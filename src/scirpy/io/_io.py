import json
import os.path
import pickle
import re
import sys
from collections.abc import Collection, Iterable, Sequence
from glob import iglob
from importlib.metadata import version
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
from anndata import AnnData

from scirpy.util import DataHandler, _doc_params, _is_na2, _is_true, _is_true2, _translate_dna_to_protein

from . import _tracerlib
from ._convert_anndata import from_airr_cells, to_airr_cells
from ._datastructures import AirrCell
from ._util import _IOLogger, _read_airr_rearrangement_df, doc_airr_fields, doc_working_model, get_rearrangement_schema

# patch sys.modules to enable pickle import.
# see https://stackoverflow.com/questions/2121874/python-pckling-after-changing-a-modules-directory
sys.modules["tracerlib"] = _tracerlib

DEFAULT_AIRR_CELL_ATTRIBUTES = "is_cell"


def _normalize_cell_attributes(cell_attributes: Collection[str]) -> tuple[str, ...]:
    if isinstance(cell_attributes, str):
        return (cell_attributes,)
    return tuple(cell_attributes)


def _airr_bool(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    value = str(value).lower()
    if value in ("t", "true"):
        return True
    if value in ("f", "false"):
        return False
    return value


def _convert_airr_dataframe_types(
    df: pd.DataFrame, *, normalize_missing_strings: bool = True, profile_read_airr: bool = False
) -> pd.DataFrame:
    import time

    total_start = time.time()
    schema_start = time.time()
    schema = get_rearrangement_schema()
    schema_end = time.time()

    replace_start = time.time()
    replaced_missing_columns = 0
    if normalize_missing_strings:
        missing_strings = ["", "NaN", "nan", "None", "N/A"]
        string_columns = df.select_dtypes(include=["object", "string"]).columns
        for column in string_columns:
            missing_mask = df[column].isin(missing_strings)
            if missing_mask.any():
                replaced_missing_columns += 1
                df.loc[missing_mask, column] = None
    replace_end = time.time()

    typed_columns = 0
    unknown_columns = 0
    boolean_columns = 0
    numeric_columns = 0
    failed_numeric_columns = 0
    column_timings = []
    for column in df.columns:
        column_start = time.time()
        try:
            field_type = schema.type(column)
        except KeyError:
            unknown_columns += 1
            continue

        typed_columns += 1
        if field_type == "boolean":
            boolean_columns += 1
            mapped = df[column].map(
                {
                    "T": True,
                    "F": False,
                    "true": True,
                    "false": False,
                    "True": True,
                    "False": False,
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                }
            )
            df[column] = mapped.where(mapped.notna(), df[column])
        elif field_type in {"integer", "number"}:
            numeric_columns += 1
            try:
                df[column] = pd.to_numeric(df[column], errors="raise")
            except (TypeError, ValueError):
                failed_numeric_columns += 1
                pass
        column_end = time.time()
        if profile_read_airr:
            column_timings.append((column, field_type, column_end - column_start))

    if profile_read_airr:
        total_end = time.time()
        print("AIRR dataframe type conversion profile:")
        print(f"  total: {total_end - total_start:.3f} s")
        print(f"  load schema: {schema_end - schema_start:.3f} s")
        print(f"  replace missing-value strings: {replace_end - replace_start:.3f} s")
        print(f"  columns with missing-value strings replaced: {replaced_missing_columns}")
        print(f"  AIRR schema columns: {typed_columns}")
        print(f"  unknown columns skipped: {unknown_columns}")
        print(f"  boolean columns converted: {boolean_columns}")
        print(f"  numeric columns converted: {numeric_columns}")
        print(f"  numeric columns left unchanged after failure: {failed_numeric_columns}")
        print("  slowest converted columns:")
        for column, field_type, elapsed in sorted(column_timings, key=lambda x: x[2], reverse=True)[:15]:
            print(f"    {column} ({field_type}): {elapsed:.3f} s")

    return df


def _series_to_awkward_array(series: pd.Series) -> ak.Array:
    if series.hasnans:
        series = series.astype(object).where(pd.notna(series), None)
        return ak.Array(series.tolist())
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        return ak.Array(series.tolist())
    return ak.Array(series.to_numpy())


def _airr_dataframe_to_columnar_chains(df: pd.DataFrame, chain_columns: Sequence[str]) -> ak.Array:
    chain_arrays = {column: _series_to_awkward_array(df[column]) for column in chain_columns}
    chain_records = ak.zip(chain_arrays, depth_limit=1)
    return ak.unflatten(chain_records, np.ones(len(df), dtype=np.int64))


def _read_airr_rearrangement_df_fast(
    path: pd.DataFrame | Sequence[pd.DataFrame],
    *,
    infer_locus: bool = True,
    cell_attributes: Collection[str] = DEFAULT_AIRR_CELL_ATTRIBUTES,
    airr_fields: Collection[str] | None = None,
    key_added: str = "airr",
    normalize_missing_strings: bool = True,
    profile_read_airr: bool = False,
) -> AnnData:
    import time

    start = time.time()
    if isinstance(path, pd.DataFrame):
        df = path
    else:
        df = pd.concat(path, ignore_index=True)

    if "cell_id" not in df.columns:
        raise ValueError("AIRR data frame must contain a `cell_id` column.")

    # Work on a shallow copy so we can add missing AIRR-required fields without modifying user data.
    df = df.copy(deep=False)
    cell_attribute_fields = _normalize_cell_attributes(cell_attributes)

    select_start = time.time()
    if airr_fields is not None:
        airr_fields = tuple(dict.fromkeys(airr_fields))
        for field in airr_fields:
            if field not in df.columns:
                df[field] = None
        selected_columns = ["cell_id"]
        selected_columns.extend(field for field in cell_attribute_fields if field in df.columns)
        selected_columns.extend(field for field in airr_fields if field != "cell_id")
        df = df.loc[:, list(dict.fromkeys(selected_columns))]
    select_end = time.time()

    required_start = time.time()
    if airr_fields is None:
        for field in get_rearrangement_schema().required:
            if field not in df.columns:
                df[field] = None
    required_end = time.time()

    if infer_locus and "locus" not in df.columns:
        infer_start = time.time()
        df["locus"] = df.apply(_infer_locus_from_gene_names, axis=1)
        infer_end = time.time()
    else:
        infer_start = infer_end = time.time()

    convert_start = time.time()
    df = _convert_airr_dataframe_types(
        df, normalize_missing_strings=normalize_missing_strings, profile_read_airr=profile_read_airr
    )
    convert_end = time.time()

    sanitize_start = sanitize_end = time.time()

    cell_id = df["cell_id"].astype(str)
    cell_attribute_fields = tuple(field for field in cell_attribute_fields if field in df.columns)

    obs_start = time.time()
    obs_columns = ["cell_id", *cell_attribute_fields]
    if cell_attribute_fields:
        duplicated_attrs = df.groupby("cell_id", sort=False)[list(cell_attribute_fields)].nunique(dropna=False)
        conflicting_attrs = duplicated_attrs.gt(1).any(axis=1)
        if conflicting_attrs.any():
            raise ValueError("Cell-level attributes differ between different chains.")
    obs = df.loc[:, obs_columns].drop_duplicates("cell_id").set_index("cell_id")
    obs.index = obs.index.astype(str)
    obs_end = time.time()

    chains_start = time.time()
    chain_columns = [column for column in df.columns if column not in {"cell_id", *cell_attribute_fields}]

    if cell_id.is_unique:
        chains = _airr_dataframe_to_columnar_chains(df, chain_columns)
    else:
        chain_df = df.loc[:, chain_columns].astype(object).where(pd.notna(df.loc[:, chain_columns]), None)
        chain_records = chain_df.to_dict(orient="records")
        chains_by_cell = {}
        ordered_cell_ids = []
        for tmp_cell_id, chain_record in zip(cell_id, chain_records, strict=False):
            try:
                tmp_chains = chains_by_cell[tmp_cell_id]
            except KeyError:
                tmp_chains = []
                chains_by_cell[tmp_cell_id] = tmp_chains
                ordered_cell_ids.append(tmp_cell_id)
            tmp_chains.append(chain_record)
        chains = [chains_by_cell[tmp_cell_id] for tmp_cell_id in ordered_cell_ids]
        chains = ak.Array(chains)
    chains_end = time.time()

    anndata_start = time.time()
    adata = AnnData(
        X=None,
        obs=obs,
        obsm={key_added: chains},
        uns={"scirpy_version": version("scirpy")},
    )
    anndata_end = time.time()

    if profile_read_airr:
        print("fast dataframe read_airr timing profile:")
        print(f"  total: {anndata_end - start:.3f} s")
        print(f"  select AIRR fields: {select_end - select_start:.3f} s")
        print(f"  add missing required fields: {required_end - required_start:.3f} s")
        print(f"  infer locus: {infer_end - infer_start:.3f} s")
        print(f"  convert AIRR field types: {convert_end - convert_start:.3f} s")
        print(f"  sanitize missing values: {sanitize_end - sanitize_start:.3f} s")
        print(f"  build obs: {obs_end - obs_start:.3f} s")
        print(f"  build chains: {chains_end - chains_start:.3f} s")
        print(f"  build anndata: {anndata_end - anndata_start:.3f} s")
        print(f"  rearrangements processed: {len(df)}")
        print(f"  cells created: {adata.n_obs}")

    return adata


def _cdr3_from_junction(junction_aa, junction_nt):
    """CDR3 euqals junction without the conserved residues C and W/F, respectively.
    Should the conserved residues not equal to C and W/F, then the chain
    is non-productive and we set CDR3 to None.

    See also https://github.com/scverse/scirpy/pull/290.
    """
    cdr3_aa, cdr3_nt = None, None
    if not _is_na2(junction_aa) and junction_aa[0] == "C" and junction_aa[-1] in ("W", "F"):
        cdr3_aa = junction_aa[1:-1]
    if (
        not _is_na2(junction_nt)
        and _translate_dna_to_protein(junction_nt[:3]) == "C"
        and _translate_dna_to_protein(junction_nt[-3:]) in ("W", "F")
    ):
        cdr3_nt = junction_nt[3:-3]
    return cdr3_aa, cdr3_nt


def _read_10x_vdj_json(
    path: str | Path,
    filtered: bool = True,
) -> Iterable[AirrCell]:
    """Read IR data from a 10x genomics `all_contig_annotations.json` file"""
    logger = _IOLogger()
    with open(path) as f:
        contigs = json.load(f)

    airr_cells: dict[str, AirrCell] = {}
    for contig in contigs:
        if filtered and not (contig["is_cell"] and contig["high_confidence"]):
            continue
        barcode = contig["barcode"]
        if barcode not in airr_cells:
            cell = AirrCell(
                barcode,
                logger=logger,
                cell_attribute_fields=("is_cell"),
            )
            airr_cells[barcode] = cell
        else:
            cell = airr_cells[barcode]

        # AIRR-compliant chain dict
        chain = AirrCell.empty_chain_dict()

        genes = {}
        mapping = {
            "L-REGION+V-REGION": "v",
            "D-REGION": "d",
            "J-REGION": "j",
            "C-REGION": "c",
        }
        for annot in contig["annotations"]:
            feat = annot["feature"]
            if feat["region_type"] in mapping:
                region = mapping[feat["region_type"]]
                assert region not in genes, region
                genes[region] = {}
                genes[region]["chain"] = feat["chain"]
                genes[region]["gene"] = feat["gene_name"]
                genes[region]["start"] = annot["contig_match_start"]
                genes[region]["end"] = annot["contig_match_end"]

        chain["v_call"] = genes["v"]["gene"] if "v" in genes else None
        chain["d_call"] = genes["d"]["gene"] if "d" in genes else None
        chain["j_call"] = genes["j"]["gene"] if "j" in genes else None
        chain["c_call"] = genes["c"]["gene"] if "c" in genes else None

        # check if chain type is consistent
        chain_types = [g["chain"] for g in genes.values()]
        chain_type = chain_types[0] if np.unique(chain_types).size == 1 else None

        # compute inserted nucleotides
        # VJ junction for TRA, TRG, IGK, IGL chains
        # VD + DJ junction for TRB, TRD, IGH chains
        #
        # Notes on indexing:
        # some tryouts have shown, that the indexes in the json file
        # seem to be python-type indexes (i.e. the 'end' index is exclusive).
        # Therefore, no `-1` needs to be subtracted when computing the number
        # of inserted nucleotides.
        chain["np1_length"] = None
        chain["np2_length"] = None
        if chain_type in AirrCell.VJ_LOCI and chain["v_call"] is not None and chain["j_call"] is not None:
            assert chain["d_call"] is None, "TRA, TRG or IG-light chains should not have a D region"
            chain["np1_length"] = genes["j"]["start"] - genes["v"]["end"]
        elif (
            chain_type in AirrCell.VDJ_LOCI
            and chain["v_call"] is not None
            and chain["j_call"] is not None
            and chain["d_call"] is not None
        ):
            chain["np1_length"] = genes["d"]["start"] - genes["v"]["end"]
            chain["np2_length"] = genes["j"]["start"] - genes["d"]["end"]

        chain["locus"] = chain_type
        chain["junction"] = contig["cdr3_seq"]
        chain["junction_aa"] = contig["cdr3"]
        chain["umi_count"] = contig["umi_count"]
        chain["consensus_count"] = contig["read_count"]
        chain["productive"] = contig["productive"]
        chain["is_cell"] = contig["is_cell"]
        chain["high_confidence"] = contig["high_confidence"]
        chain["sequence"] = contig["sequence"]
        chain["sequence_aa"] = contig["aa_sequence"]
        chain["sequence_id"] = contig["contig_name"]

        # additional cols from CR6 outputs: fwr{1,2,3,4}{,_nt} and cdr{1,2}{,_nt}
        fwrs = [f"fwr{i}" for i in range(1, 5)]
        cdrs = [f"cdr{i}" for i in range(1, 3)]

        for col in fwrs + cdrs:
            if col in contig.keys():
                chain[col] = contig[col].get("nt_seq") if contig[col] else None
                chain[col + "_aa"] = contig[col].get("aa_seq") if contig[col] else None

        chain["cdr3_aa"], chain["cdr3"] = _cdr3_from_junction(chain["junction_aa"], chain["junction"])

        cell.add_chain(chain)

    return airr_cells.values()


def _read_10x_vdj_csv(
    path: str | Path,
    filtered: bool = True,
) -> Iterable[AirrCell]:
    """Read IR data from a 10x genomics `_contig_annotations.csv` file"""
    logger = _IOLogger()
    df = pd.read_csv(path)

    airr_cells = {}
    if filtered:
        df = df.loc[_is_true(df["is_cell"]) & _is_true(df["high_confidence"]), :]
    for barcode, cell_df in df.groupby("barcode"):
        ir_obj = AirrCell(barcode, logger=logger, cell_attribute_fields=("is_cell"))
        for _, chain_series in cell_df.iterrows():
            chain_dict = AirrCell.empty_chain_dict()
            chain_dict.update(
                locus=chain_series["chain"],
                junction_aa=chain_series["cdr3"],
                junction=chain_series["cdr3_nt"],
                umi_count=chain_series["umis"],
                consensus_count=chain_series["reads"],
                productive=_is_true2(chain_series["productive"]),
                v_call=chain_series["v_gene"],
                d_call=chain_series["d_gene"],
                j_call=chain_series["j_gene"],
                c_call=chain_series["c_gene"],
                is_cell=chain_series["is_cell"],
                high_confidence=chain_series["high_confidence"],
                sequence_id=chain_series["contig_id"],
            )

            # additional cols from CR6 outputs: fwr{1,2,3,4}{,_nt} and cdr{1,2}{,_nt}
            fwrs = [f"fwr{i}" for i in range(1, 5)]
            cdrs = [f"cdr{i}" for i in range(1, 3)]

            for col in fwrs + cdrs:
                if col in chain_series.index:
                    chain_dict[col + "_aa"] = chain_series.get(col)
                if col + "_nt" in chain_series.index:
                    chain_dict[col] = chain_series.get(col + "_nt")

            chain_dict["cdr3_aa"], chain_dict["cdr3"] = _cdr3_from_junction(
                chain_dict["junction_aa"], chain_dict["junction"]
            )

            ir_obj.add_chain(chain_dict)

        airr_cells[barcode] = ir_obj

    return airr_cells.values()


@_doc_params(doc_working_model=doc_working_model)
def read_10x_vdj(path: str | Path, filtered: bool = True, include_fields: Any = None, **kwargs) -> AnnData:
    """\
    Read :term:`AIRR` data from 10x Genomics cell-ranger output.

    Supports `all_contig_annotations.json` and
    `{{all,filtered}}_contig_annotations.csv`.

    If the `json` file is available, it is preferable as it
    contains additional information about V(D)J-junction insertions. Other than
    that there should be no difference.

    {doc_working_model}

    Parameters
    ----------
    path
        Path to `filterd_contig_annotations.csv`, `all_contig_annotations.csv` or
        `all_contig_annotations.json`.
    filtered
        Only keep filtered contig annotations (i.e. `is_cell` and `high_confidence`).
        If using `filtered_contig_annotations.csv` already, this option
        is futile.
    include_fields
        Deprecated. Does not have any effect as of v0.13.
    **kwargs
        are passed to :func:`~scirpy.io.from_airr_cells`.

    Returns
    -------
    AnnData object with :term:`AIRR` data in `obsm["airr"]` for each cell. For more details see
    :ref:`data-structure`.
    """
    path = Path(path)
    if path.suffix == ".json":
        airr_cells = _read_10x_vdj_json(path, filtered)
    else:
        airr_cells = _read_10x_vdj_csv(path, filtered)

    return from_airr_cells(airr_cells, **kwargs)


@_doc_params(doc_working_model=doc_working_model)
def read_tracer(path: str | Path, **kwargs) -> AnnData:
    """\
    Read data from `TraCeR <https://github.com/Teichlab/tracer>`_ (:cite:`Stubbington2016-kh`).

    Requires the TraCeR output directory which contains a folder for each cell.
    Unfortunately the results files generated by `tracer summarize` do not
    contain all required information.

    The function will read TCR information from the `filtered_TCR_seqs/<CELL_ID>.pkl`
    files.

    {doc_working_model}

    Parameters
    ----------
    path
        Path to the TraCeR output folder.
    **kwargs
        are passed to :func:`~scirpy.io.from_airr_cells`.

    Returns
    -------
    AnnData object with :term:`AIRR` data in `obsm["airr"]` for each cell. For more details see
    :ref:`data-structure`.
    """
    logger = _IOLogger()
    airr_cells = {}
    path = str(path)

    def _process_chains(chains, chain_type):
        for tmp_chain in chains:
            if tmp_chain.cdr3 == "N/A" or tmp_chain.cdr3nt == "N/A":
                # ignore chains that have no sequence
                continue

            # AIRR-rearrangement compliant chain dictionary
            chain_dict = AirrCell.empty_chain_dict()
            if tmp_chain.has_D_segment:
                assert chain_type in AirrCell.VDJ_LOCI
                assert len(tmp_chain.junction_details) == 5
                assert len(tmp_chain.summary) == 8
                chain_dict["v_call"] = tmp_chain.summary[0].split("*")[0]
                chain_dict["d_call"] = tmp_chain.summary[1].split("*")[0]
                chain_dict["j_call"] = tmp_chain.summary[2].split("*")[0]
            else:
                assert chain_type in AirrCell.VJ_LOCI
                assert len(tmp_chain.junction_details) == 3
                assert len(tmp_chain.summary) == 7
                chain_dict["v_call"] = tmp_chain.summary[0].split("*")[0]
                chain_dict["d_call"] = None
                chain_dict["j_call"] = tmp_chain.summary[1].split("*")[0]

            for call_key in ["v_call", "d_call", "j_call"]:
                if chain_dict[call_key] == "N/A":
                    chain_dict[call_key] = None

                if chain_dict[call_key] is not None:
                    assert chain_dict[call_key][3] == call_key[0].upper()

            chain_dict["np1_length"] = (
                len(tmp_chain.junction_details[1]) if tmp_chain.junction_details[1] != "N/A" else None
            )
            try:
                # only in VDJ
                chain_dict["np2_length"] = (
                    len(tmp_chain.junction_details[3]) if tmp_chain.junction_details[3] != "N/A" else None
                )
            except IndexError:
                chain_dict["np2_length"] = None

            chain_dict["locus"] = chain_type
            chain_dict["consensus_count"] = tmp_chain.TPM
            chain_dict["productive"] = tmp_chain.productive
            chain_dict["junction"] = tmp_chain.cdr3nt
            chain_dict["junction_aa"] = tmp_chain.cdr3

            yield chain_dict

    for summary_file in iglob(os.path.join(path, "**/filtered_TCR_seqs/*.pkl"), recursive=True):
        cell_name = summary_file.split(os.sep)[-3]
        airr_cell = AirrCell(cell_name, logger=logger)
        try:
            with open(summary_file, "rb") as f:
                tracer_obj = pickle.load(f)
                chains = tracer_obj.recombinants["TCR"]
                for chain_id in "ABGD":
                    if chain_id in chains and chains[chain_id] is not None:
                        for tmp_chain in _process_chains(chains[chain_id], f"TR{chain_id}"):
                            airr_cell.add_chain(tmp_chain)
        except ImportError as e:
            # except Exception as e:
            raise Exception(f"Error loading TCR data from cell {summary_file}") from e

        airr_cells[cell_name] = airr_cell

    if not len(airr_cells):
        raise OSError(
            "Could not find any TraCeR *.pkl files. Make sure you are "
            "using a TraCeR output folder that looks like "
            "<CELL>/filtered_TCR_seqs/*.pkl"
        )

    return from_airr_cells(airr_cells.values(), **kwargs)


@_doc_params(
    doc_working_model=doc_working_model,
    doc_airr_fields=doc_airr_fields,
    cell_attributes=f"""`({",".join([f'"{x}"' for x in DEFAULT_AIRR_CELL_ATTRIBUTES])})`""",
)
def read_airr(
    path: str | Sequence[str] | Path | Sequence[Path] | pd.DataFrame | Sequence[pd.DataFrame],
    use_umi_count_col: None = None,  # deprecated, kept for backwards-compatibility
    infer_locus: bool = True,
    cell_attributes: Collection[str] = DEFAULT_AIRR_CELL_ATTRIBUTES,
    include_fields: Any = None,
    airr_fields: Collection[str] | None = None,
    normalize_missing_strings: bool = True,
    profile_read_airr: bool = False,
    **kwargs,
) -> AnnData:
    """\
    Read data from `AIRR rearrangement <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_ format.

    {doc_airr_fields}

    {doc_working_model}

    Parameters
    ----------
    path
        Path to the AIRR rearrangement tsv file. If different
        chains are split up into multiple files, these can be specified
        as a List, e.g. `["path/to/tcr_alpha.tsv", "path/to/tcr_beta.tsv"]`.
        Alternatively, this can be a pandas data frame.
    use_umi_count_col
        Deprecated, has no effect as of v0.16. Since v1.4 of the AIRR standard, `umi_count`
        is an official field in the Rearrangement schema and preferred over `duplicate_count`.
        `umi_count` now always takes precedence over `duplicate_count`.
    infer_locus
        Try to infer the `locus` column from gene names, in case it is not specified.
    cell_attributes
        Fields in the rearrangement schema that are specific for a cell rather
        than a chain. The values must be identical over all records belonging to a
        cell. This defaults to {cell_attributes}.
    include_fields
        Deprecated. Does not have any effect as of v0.13.
    airr_fields
        AIRR fields to keep when reading from a pandas data frame. By default,
        all fields are preserved.
    normalize_missing_strings
        Replace AIRR-style missing string values such as `""`, `"nan"`, and `"N/A"`
        with `None` when reading from a pandas data frame.
    profile_read_airr
        Print timing measurements for the fast pandas data frame path.
    **kwargs
        are passed to :func:`~scirpy.io.from_airr_cells`.

    Returns
    -------
    AnnData object with :term:`AIRR` data in `obsm["airr"]` for each cell. For more details see
    :ref:`data-structure`..
    """
    # defer import, as this is very slow
    import airr

    airr_cells = {}
    logger = _IOLogger()

    if isinstance(path, str | Path | pd.DataFrame):
        path: list[str | Path | pd.DataFrame] = [path]  # type: ignore

    if all(isinstance(tmp_path_or_df, pd.DataFrame) for tmp_path_or_df in path):
        key_added = kwargs.pop("key_added", "airr")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        return _read_airr_rearrangement_df_fast(
            path,  # type: ignore[arg-type]
            infer_locus=infer_locus,
            cell_attributes=cell_attributes,
            airr_fields=airr_fields,
            key_added=key_added,
            normalize_missing_strings=normalize_missing_strings,
            profile_read_airr=profile_read_airr,
        )

    for tmp_path_or_df in path:
        if isinstance(tmp_path_or_df, pd.DataFrame):
            iterator = _read_airr_rearrangement_df(tmp_path_or_df)
        else:
            iterator = airr.read_rearrangement(str(tmp_path_or_df))

        for chain_dict in iterator:
            cell_id = chain_dict.pop("cell_id")
            chain_dict.update({req: None for req in get_rearrangement_schema().required if req not in chain_dict})
            try:
                tmp_cell = airr_cells[cell_id]
            except KeyError:
                tmp_cell = AirrCell(
                    cell_id=cell_id,
                    logger=logger,
                    cell_attribute_fields=cell_attributes,
                )
                airr_cells[cell_id] = tmp_cell

            if infer_locus and "locus" not in chain_dict:
                logger.warning(
                    "`locus` column not found in input data. The locus is being inferred from the "
                    "{v,d,j,c}_call columns."
                )
                chain_dict["locus"] = _infer_locus_from_gene_names(chain_dict)

            tmp_cell.add_chain(chain_dict)

    return from_airr_cells(airr_cells.values(), **kwargs)


def _infer_locus_from_gene_names(chain_dict, *, keys=("v_call", "d_call", "j_call", "c_call")):
    """Infer the IMGT locus name from VDJ calls"""
    keys = list(keys)
    # TRAV.*/DV is misleading as it actually points to a delta locus
    # See #285
    if not _is_na2(chain_dict["v_call"]) and re.search("TRAV.*/DV", chain_dict["v_call"]):
        keys.remove("v_call")

    genes = []
    for k in keys:
        gene = chain_dict[k]
        if not _is_na2(gene):
            genes.append(gene.lower())

    if not len(genes):
        locus = None
    elif all("tra" in x for x in genes):
        locus = "TRA"
    elif all("trb" in x for x in genes):
        locus = "TRB"
    elif all("trd" in x for x in genes):
        locus = "TRD"
    elif all("trg" in x for x in genes):
        locus = "TRG"
    elif all("igh" in x for x in genes):
        locus = "IGH"
    elif all("igk" in x for x in genes):
        locus = "IGK"
    elif all("igl" in x for x in genes):
        locus = "IGL"
    else:
        locus = None

    return locus


@_doc_params(doc_working_model=doc_working_model)
def read_bracer(path: str | Path, **kwargs) -> AnnData:
    """\
    Read data from `BraCeR <https://github.com/Teichlab/bracer>`_ (:cite:`Lindeman2018`).

    Requires the `changeodb.tab` file as input which is created by the
    `bracer summarise` step.

    {doc_working_model}

    Parameters
    ----------
    path
        Path to the `changeodb.tab` file.
    **kwargs
        are passed to :func:`~scirpy.io.from_airr_cells`.

    Returns
    -------
    AnnData object with :term:`AIRR` data in `obsm["airr"]` for each cell. For more details see
    :ref:`data-structure`.
    """
    logger = _IOLogger()
    changeodb = pd.read_csv(path, sep="\t", na_values=["None"])

    bcr_cells = {}
    for _, row in changeodb.iterrows():
        cell_id = row["CELL"]
        try:
            tmp_ir_cell = bcr_cells[cell_id]
        except KeyError:
            tmp_ir_cell = AirrCell(cell_id, logger=logger)
            bcr_cells[cell_id] = tmp_ir_cell

        chain_dict = AirrCell.empty_chain_dict()

        chain_dict["v_call"] = row["V_CALL"] if not pd.isnull(row["V_CALL"]) else None
        chain_dict["d_call"] = row["D_CALL"] if not pd.isnull(row["D_CALL"]) else None
        chain_dict["j_call"] = row["J_CALL"] if not pd.isnull(row["J_CALL"]) else None
        chain_dict["c_call"] = row["C_CALL"].split("*")[0] if not pd.isnull(row["C_CALL"]) else None
        chain_dict["locus"] = "IG" + row["LOCUS"]

        chain_dict["np1_length"] = None
        chain_dict["np2_length"] = None
        if (
            chain_dict["locus"] in AirrCell.VJ_LOCI
            and not pd.isnull(row["V_SEQ_START"])
            and not pd.isnull(row["J_SEQ_START"])
        ):
            assert pd.isnull(row["D_SEQ_START"]), "TRA, TRG or IG-light chains should not have a D region" + str(row)
            chain_dict["np1_length"] = row["J_SEQ_START"] - (row["V_SEQ_START"] + row["V_SEQ_LENGTH"])  # type: ignore
        elif (
            chain_dict["locus"] in AirrCell.VDJ_LOCI
            and not pd.isnull(row["V_SEQ_START"])
            and not pd.isnull(row["D_SEQ_START"])
            and not pd.isnull(row["J_SEQ_START"])
        ):
            chain_dict["np1_length"] = row["D_SEQ_START"] - (row["V_SEQ_START"] + row["V_SEQ_LENGTH"])  # type: ignore
            chain_dict["np2_length"] = row["J_SEQ_START"] - (row["D_SEQ_START"] + row["D_SEQ_LENGTH"])  # type: ignore

        chain_dict["junction"] = row["JUNCTION"] if not pd.isnull(row["JUNCTION"]) else None
        chain_dict["junction_aa"] = (
            _translate_dna_to_protein(chain_dict["junction"]) if chain_dict["junction"] is not None else None
        )
        chain_dict["consensus_count"] = row["TPM"]
        chain_dict["productive"] = row["FUNCTIONAL"]

        tmp_ir_cell.add_chain(chain_dict)

    return from_airr_cells(bcr_cells.values(), **kwargs)


def write_airr(adata: DataHandler.TYPE, filename: str | Path, **kwargs) -> None:
    """Export :term:`IR` data to :term:`AIRR` Rearrangement `tsv` format.

    Parameters
    ----------
    adata
        annotated data matrix
    filename
        destination filename
    **kwargs
        additional arguments passed to :func:`~scirpy.io.to_airr_cells`
    """
    # defer import, as this is very slow
    import airr

    airr_cells = to_airr_cells(adata, **kwargs)
    try:
        fields = airr_cells[0].fields
        for tmp_cell in airr_cells[1:]:
            assert tmp_cell.fields == fields, "All rows of adata have the same fields."
    except IndexError:
        # case of an empty output file
        fields = None

    writer = airr.create_rearrangement(filename, fields=fields)
    for tmp_cell in airr_cells:
        for chain in tmp_cell.to_airr_records():
            # workaround for AIRR library writing out int field as floats (if it happens to be a float)
            for field, value in chain.items():
                if airr.RearrangementSchema.type(field) == "integer" and value is not None:
                    chain[field] = int(value)
            writer.write(chain)
    writer.close()


def to_dandelion(adata: DataHandler.TYPE):
    """Export data to `Dandelion <https://github.com/zktuong/dandelion>`_ (:cite:`Stephenson2021`).

    Parameters
    ----------
    adata
        annotated data matrix with :term:`IR` annotations.

    Returns
    -------
    `Dandelion` object.
    """
    try:
        try:
            from dandelion import from_scirpy
        except ImportError:
            # moved in dandelion 1.0 pre-release
            from dandelion.tools import from_scirpy
    except ImportError:
        raise ImportError("Please install dandelion: pip install sc-dandelion.") from None

    return from_scirpy(adata)


@_doc_params(doc_working_model=doc_working_model)
def from_dandelion(dandelion, transfer: bool = False, to_mudata: bool = False, **kwargs) -> AnnData:
    """\
    Import data from `Dandelion <https://github.com/zktuong/dandelion>`_ (:cite:`Stephenson2021`).

    Internally calls `dandelion.to_scirpy`.

    {doc_working_model}

    Parameters
    ----------
    dandelion
        a `dandelion.Dandelion` instance
    transfer
        Whether to execute `dandelion.tl.transfer` to transfer all data
        to the :class:`anndata.AnnData` instance.
    to_mudata
        Return MuData object instead of AnnData object.
    **kwargs
        Additional arguments passed to `dandelion.to_scirpy`.

    Returns
    -------
    AnnData object with :term:`AIRR` data in `obsm["airr"]` for each cell. For more details see
    :ref:`data-structure`.
    """
    try:
        try:
            from dandelion import to_scirpy
        except ImportError:
            # moved in dandelion 1.0 pre-release
            from dandelion.tools import to_scirpy
    except ImportError:
        raise ImportError("Please install dandelion: pip install sc-dandelion.") from None

    return to_scirpy(dandelion, transfer=transfer, to_mudata=to_mudata, **kwargs)


@_doc_params(doc_working_model=doc_working_model)
def read_bd_rhapsody(path: str | Path, **kwargs) -> AnnData:
    """\
    Read :term:`IR` data from the BD Rhapsody Analysis Pipeline.

    Supports `*_perCellChain.csv`, `*_perCellChain_unfiltered.csv`, `*_VDJ_Dominant_Contigs.csv`, and
    `*_VDJ_Unfiltered_Contigs.csv` files. The applicable filename depends your version of the BD Rhapsody pipeline.

    .. note::

        More recent versions of the pipeline generate data in standardized `AIRR Rearragement format <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.
        If you have a chance to do so, we recommend reanalysing your data with the most recent version of the
        BD Rhapsody pipeline and read output filese with :func:`scirpy.io.read_airr`.

    `*_perCell` files are currently not supported, follow the :ref:`IO Tutorial <importing-custom-formats>` to import
    custom formats and make use of `this snippet <https://github.com/scverse/scirpy/blob/8de293fd54125a00f8ff6fd5f8a4cb232d7a51e6/scirpy/io/_io_bdrhapsody.py#L8-L44>`_

    {doc_working_model}

    Parameters
    ----------
    path:
        Path to the `perCellChain` or `Contigs` file generated by the BD Rhapsody analysis pipeline. May be gzipped.
    **kwargs
        are passed to :func:`~scirpy.io.from_airr_cells`.

    Returns
    -------
    AnnData object with :term:`AIRR` data in `obsm["airr"]` for each cell. For more details see
    :ref:`data-structure`.
    """
    df = pd.read_csv(path, comment="#", index_col=0)

    def _translate_locus(locus):
        return {
            "TCR_Alpha": "TRA",
            "TCR_Beta": "TRB",
            "TCR_Gamma": "TRG",
            "TCR_Delta": "TRD",
            "BCR_Lambda": "IGL",
            "BCR_Kappa": "IGK",
            "BCR_Heavy": "IGH",
        }[locus]

    def _get(row, field):
        try:
            return row[field]
        except KeyError:
            # in "Dominant_Contigs", the fields end with "_Dominant"
            return row[f"{field}_Dominant"]

    airr_cells = {}
    for idx, row in df.iterrows():
        idx = str(idx)
        if idx not in airr_cells:
            airr_cells[idx] = AirrCell(cell_id=idx)
        tmp_cell = airr_cells[idx]
        tmp_chain = AirrCell.empty_chain_dict()
        tmp_chain.update(
            {
                "locus": _translate_locus(row["Chain_Type"]),
                "v_call": _get(row, "V_gene"),
                "d_call": _get(row, "D_gene"),
                "j_call": _get(row, "J_gene"),
                "c_call": _get(row, "C_gene"),
                # strictly, this would have to go the the "cdr3" field as the conserved residues
                # are not contained. However we only work with `junction` in scirpy
                "junction": _get(row, "CDR3_Nucleotide"),
                "junction_aa": _get(row, "CDR3_Translation"),
                "productive": row["Productive"],
                "consensus_count": row["Read_Count"],
                "umi_count": row["Molecule_Count"],
            }
        )
        tmp_cell.add_chain(tmp_chain)

    return from_airr_cells(airr_cells.values(), **kwargs)

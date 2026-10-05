"""Read one source-spec branch from an XLSX workbook without modifying it."""

from __future__ import annotations

import hashlib
import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree

from .source_contracts import (
    SourceContractError,
    make_source_spec_record_v2,
    validate_selected_spec_ref,
)


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOCUMENT_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

CORE_SOURCE_COLUMNS = (
    "spec_id",
    "branch_id",
    "rule_text",
    "given",
    "when",
    "then",
    "deontic",
)

SOURCE_METADATA_COLUMNS = (
    "kind",
    "origin",
    "evidence_quote",
)

REQUIRED_SOURCE_COLUMNS = CORE_SOURCE_COLUMNS + SOURCE_METADATA_COLUMNS

LITERAL_SOURCE_COLUMNS = REQUIRED_SOURCE_COLUMNS + (
    "review_ok",
    "review_note",
)

CELL_REFERENCE_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


class SourceAnchorLoadError(SourceContractError):
    """Raised when a SelectedSpecRef cannot be loaded exactly from XLSX."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceAnchorLoadError(f"cannot read workbook {path}: {exc}") from exc
    return digest.hexdigest()


def _column_index(reference: str) -> int:
    match = CELL_REFERENCE_RE.fullmatch(reference)
    if match is None:
        raise SourceAnchorLoadError(f"invalid XLSX cell reference: {reference!r}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _zip_member_path(target: str) -> str:
    if target.startswith("/"):
        candidate = target.lstrip("/")
    else:
        candidate = posixpath.join("xl", target)
    normalized = posixpath.normpath(candidate)
    if normalized == ".." or normalized.startswith("../"):
        raise SourceAnchorLoadError(
            f"workbook relationship escapes its XLSX package: {target!r}"
        )
    return normalized


def _xml_root(archive: zipfile.ZipFile, member: str) -> ElementTree.Element:
    try:
        raw = archive.read(member)
    except KeyError as exc:
        raise SourceAnchorLoadError(
            f"workbook is missing required XLSX member {member!r}"
        ) from exc
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise SourceAnchorLoadError(
            f"XLSX member {member!r} is not valid XML: {exc}"
        ) from exc


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = _xml_root(archive, "xl/sharedStrings.xml")
    values = []
    for item in root.findall(f"{{{SPREADSHEET_NS}}}si"):
        values.append(
            "".join(
                node.text or ""
                for node in item.iter(f"{{{SPREADSHEET_NS}}}t")
            )
        )
    return values


def _worksheet_member(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = _xml_root(archive, "xl/workbook.xml")
    relationships = _xml_root(archive, "xl/_rels/workbook.xml.rels")
    targets = {
        relationship.get("Id"): relationship.get("Target")
        for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        if relationship.get("TargetMode") != "External"
    }
    available = []
    for sheet in workbook.findall(
        f".//{{{SPREADSHEET_NS}}}sheets/{{{SPREADSHEET_NS}}}sheet"
    ):
        name = sheet.get("name")
        if name:
            available.append(name)
        if name != sheet_name:
            continue
        relationship_id = sheet.get(f"{{{DOCUMENT_REL_NS}}}id")
        target = targets.get(relationship_id)
        if not target:
            raise SourceAnchorLoadError(
                f"sheet {sheet_name!r} has no internal worksheet relationship"
            )
        return _zip_member_path(target)
    raise SourceAnchorLoadError(
        f"sheet {sheet_name!r} does not exist; available sheets: {available!r}"
    )


def _cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
) -> tuple[Any, bool]:
    has_formula = cell.find(f"{{{SPREADSHEET_NS}}}f") is not None
    value_type = cell.get("t")
    if value_type == "inlineStr":
        text = "".join(
            node.text or ""
            for node in cell.iter(f"{{{SPREADSHEET_NS}}}t")
        )
        return text, has_formula

    value_node = cell.find(f"{{{SPREADSHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None, has_formula
    raw = value_node.text
    if value_type == "s":
        try:
            return shared_strings[int(raw)], has_formula
        except (ValueError, IndexError) as exc:
            raise SourceAnchorLoadError(
                f"invalid shared string index in cell {cell.get('r')!r}: {raw!r}"
            ) from exc
    if value_type in {"str", "e"}:
        return raw, has_formula
    if value_type == "b":
        if raw not in {"0", "1"}:
            raise SourceAnchorLoadError(
                f"invalid boolean value in cell {cell.get('r')!r}: {raw!r}"
            )
        return raw == "1", has_formula
    try:
        number = float(raw)
    except ValueError:
        return raw, has_formula
    if number.is_integer():
        return int(number), has_formula
    return number, has_formula


def _read_rows(
    archive: zipfile.ZipFile,
    worksheet_member: str,
    shared_strings: list[str],
    wanted_rows: set[int],
) -> dict[int, dict[int, tuple[Any, bool]]]:
    worksheet = _xml_root(archive, worksheet_member)
    result: dict[int, dict[int, tuple[Any, bool]]] = {}
    for row in worksheet.findall(
        f".//{{{SPREADSHEET_NS}}}sheetData/{{{SPREADSHEET_NS}}}row"
    ):
        raw_row_number = row.get("r")
        try:
            row_number = int(raw_row_number or "")
        except ValueError as exc:
            raise SourceAnchorLoadError(
                f"invalid XLSX row number: {raw_row_number!r}"
            ) from exc
        if row_number not in wanted_rows:
            continue
        cells: dict[int, tuple[Any, bool]] = {}
        for cell in row.findall(f"{{{SPREADSHEET_NS}}}c"):
            reference = cell.get("r")
            if not reference:
                raise SourceAnchorLoadError(
                    f"cell in row {row_number} has no XLSX reference"
                )
            column = _column_index(reference)
            if column in cells:
                raise SourceAnchorLoadError(
                    f"row {row_number} contains duplicate column index {column}"
                )
            cells[column] = _cell_value(cell, shared_strings)
        result[row_number] = cells
    return result


def _row_numbers(
    archive: zipfile.ZipFile,
    worksheet_member: str,
) -> set[int]:
    worksheet = _xml_root(archive, worksheet_member)
    result = set()
    for row in worksheet.findall(
        f".//{{{SPREADSHEET_NS}}}sheetData/{{{SPREADSHEET_NS}}}row"
    ):
        raw = row.get("r")
        try:
            result.add(int(raw or ""))
        except ValueError as exc:
            raise SourceAnchorLoadError(
                f"invalid XLSX row number: {raw!r}"
            ) from exc
    return result


def _headers(cells: Mapping[int, tuple[Any, bool]], header_row: int) -> list[str]:
    if not cells:
        raise SourceAnchorLoadError(f"header row {header_row} is empty")
    last_column = max(cells)
    headers = []
    for column in range(last_column + 1):
        value, has_formula = cells.get(column, (None, False))
        if has_formula:
            raise SourceAnchorLoadError(
                f"header row {header_row}, column {column + 1} must not contain a formula"
            )
        if not isinstance(value, str) or not value.strip():
            raise SourceAnchorLoadError(
                f"header row {header_row}, column {column + 1} must contain a name"
            )
        headers.append(value)
    duplicates = sorted({value for value in headers if headers.count(value) > 1})
    if duplicates:
        raise SourceAnchorLoadError(
            f"workbook contains duplicate source headers: {duplicates!r}"
        )
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(headers))
    if missing:
        raise SourceAnchorLoadError(
            f"workbook is missing required source columns: {missing!r}"
        )
    return headers


def load_source_spec_record(
    selected_spec_ref: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    header_row: int = 1,
) -> dict[str, Any]:
    """Load exactly one Excel branch and return a fingerprinted SourceSpecRecord.

    Relative workbook locators are resolved against ``base_dir``.  The function
    is intentionally read-only and rejects formula-derived core source fields.
    """

    ref = validate_selected_spec_ref(selected_spec_ref)
    if not isinstance(header_row, int) or isinstance(header_row, bool) or header_row < 1:
        raise SourceAnchorLoadError("header_row must be a positive integer")
    if ref["row"] <= header_row:
        raise SourceAnchorLoadError(
            f"selected row {ref['row']} must be after header row {header_row}"
        )
    workbook_path = Path(ref["workbook"])
    if not workbook_path.is_absolute():
        root = Path.cwd() if base_dir is None else Path(base_dir)
        workbook_path = root / workbook_path
    workbook_path = workbook_path.resolve()
    if not workbook_path.is_file():
        raise SourceAnchorLoadError(f"workbook does not exist: {workbook_path}")

    workbook_hash = _file_sha256(workbook_path)
    try:
        archive_context = zipfile.ZipFile(workbook_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceAnchorLoadError(
            f"workbook is not a readable XLSX package: {workbook_path}: {exc}"
        ) from exc
    with archive_context as archive:
        worksheet_member = _worksheet_member(archive, ref["sheet"])
        strings = _shared_strings(archive)
        rows = _read_rows(
            archive,
            worksheet_member,
            strings,
            {header_row, ref["row"]},
        )

    if header_row not in rows:
        raise SourceAnchorLoadError(f"header row {header_row} does not exist")
    if ref["row"] not in rows:
        raise SourceAnchorLoadError(f"selected row {ref['row']} does not exist")
    headers = _headers(rows[header_row], header_row)
    values: dict[str, Any] = {}
    formula_headers = []
    target_cells = rows[ref["row"]]
    for column, header in enumerate(headers):
        value, has_formula = target_cells.get(column, (None, False))
        values[header] = value
        if has_formula:
            formula_headers.append(header)
    forbidden_formulas = sorted(set(formula_headers) & set(LITERAL_SOURCE_COLUMNS))
    if forbidden_formulas:
        raise SourceAnchorLoadError(
            "source anchor columns must contain literal values, not formulas: "
            f"{forbidden_formulas!r}"
        )
    try:
        return make_source_spec_record_v2(
            selected_spec_ref=ref,
            spec_id=values.get("spec_id"),
            branch_id=values.get("branch_id"),
            source_kind=values.get("kind"),
            source_origin=values.get("origin"),
            rule_text=values.get("rule_text"),
            given=values.get("given"),
            when=values.get("when"),
            then=values.get("then"),
            deontic=values.get("deontic"),
            evidence_quote=values.get("evidence_quote") or None,
            review_status=values.get("review_ok"),
            review_note=values.get("review_note"),
            source_columns=values,
            loader_provenance={
                "adapter_id": "xlsx-tabular-source-anchor/v0.2",
                "workbook_sha256": workbook_hash,
                "sheet": ref["sheet"],
                "row": ref["row"],
                "header_row": header_row,
                "headers": headers,
                "formula_columns": formula_headers,
            },
        )
    except SourceContractError as exc:
        raise SourceAnchorLoadError(
            f"selected source row {ref['row']} violates SourceSpecRecord: {exc}"
        ) from exc


def list_selected_spec_refs(
    *,
    workbook: str | Path,
    sheet: str,
    base_dir: str | Path | None = None,
    header_row: int = 1,
    review_ok: int | None = None,
) -> list[dict[str, Any]]:
    """List source locators for a worksheet without interpreting spec semantics."""

    if not isinstance(sheet, str) or not sheet.strip():
        raise SourceAnchorLoadError("sheet must be a non-empty string")
    if not isinstance(header_row, int) or isinstance(header_row, bool) or header_row < 1:
        raise SourceAnchorLoadError("header_row must be a positive integer")
    workbook_locator = str(workbook)
    workbook_path = Path(workbook_locator)
    if not workbook_path.is_absolute():
        root = Path.cwd() if base_dir is None else Path(base_dir)
        workbook_path = root / workbook_path
    workbook_path = workbook_path.resolve()
    if not workbook_path.is_file():
        raise SourceAnchorLoadError(f"workbook does not exist: {workbook_path}")

    try:
        archive_context = zipfile.ZipFile(workbook_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceAnchorLoadError(
            f"workbook is not a readable XLSX package: {workbook_path}: {exc}"
        ) from exc
    with archive_context as archive:
        member = _worksheet_member(archive, sheet)
        row_numbers = _row_numbers(archive, member)
        rows = _read_rows(
            archive,
            member,
            _shared_strings(archive),
            row_numbers,
        )
    if header_row not in rows:
        raise SourceAnchorLoadError(f"header row {header_row} does not exist")
    headers = _headers(rows[header_row], header_row)
    branch_column = headers.index("branch_id")
    review_column = headers.index("review_ok") if "review_ok" in headers else None
    if review_ok is not None and review_column is None:
        raise SourceAnchorLoadError(
            "review_ok filter was requested but the worksheet has no review_ok column"
        )

    result = []
    for row_number in sorted(value for value in rows if value > header_row):
        branch_id, branch_formula = rows[row_number].get(
            branch_column, (None, False)
        )
        if branch_id in {None, ""}:
            continue
        if branch_formula:
            raise SourceAnchorLoadError(
                f"branch_id in row {row_number} must be a literal value"
            )
        if review_ok is not None:
            actual_review, review_formula = (
                rows[row_number].get(review_column, (None, False))
                if review_column is not None
                else (None, False)
            )
            if review_formula:
                raise SourceAnchorLoadError(
                    f"review_ok in row {row_number} must be a literal value"
                )
            if actual_review != review_ok:
                continue
        try:
            result.append(
                validate_selected_spec_ref(
                    {
                        "schema_version": "agentspectesting.selected-spec-ref/v0.1",
                        "workbook": workbook_locator,
                        "sheet": sheet,
                        "row": row_number,
                        "branch_id": branch_id,
                    }
                )
            )
        except SourceContractError as exc:
            raise SourceAnchorLoadError(
                f"row {row_number} cannot form a SelectedSpecRef: {exc}"
            ) from exc
    return result


__all__ = [
    "CORE_SOURCE_COLUMNS",
    "LITERAL_SOURCE_COLUMNS",
    "REQUIRED_SOURCE_COLUMNS",
    "SOURCE_METADATA_COLUMNS",
    "SourceAnchorLoadError",
    "list_selected_spec_refs",
    "load_source_spec_record",
]

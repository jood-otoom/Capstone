from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import IMAGE_EXTENSIONS, detect_project_root, read_yaml_file, to_float, write_csv


@dataclass
class SplitHealth:
    dataset_name: str
    split: str
    image_count: int
    label_count: int
    missing_labels: list[Path]
    extra_labels: list[Path]
    empty_labels: list[Path]
    labels_expected: bool


def collect_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes])


def parse_class_info(dataset_yaml: Path) -> tuple[int | None, list[str]]:
    yaml_data = read_yaml_file(dataset_yaml)
    nc_raw = yaml_data.get("nc")
    nc = int(nc_raw) if nc_raw is not None else None
    names_raw = yaml_data.get("names")

    if isinstance(names_raw, dict):
        names = [str(names_raw[key]) for key in sorted(names_raw.keys(), key=lambda value: int(value))]
    elif isinstance(names_raw, list):
        names = [str(name) for name in names_raw]
    elif isinstance(names_raw, str):
        cleaned = names_raw.strip().strip("[]")
        names = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
    else:
        names = []

    if nc is None and names:
        nc = len(names)
    return nc, names


def inspect_split(dataset_name: str, dataset_dir: Path, split: str, labels_expected: bool) -> SplitHealth:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    image_files = collect_files(images_dir, IMAGE_EXTENSIONS)
    label_files = collect_files(labels_dir, {".txt"})

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    missing_stems = sorted(image_stems - label_stems) if labels_expected else []
    extra_stems = sorted(label_stems - image_stems)

    missing_labels = [labels_dir / f"{stem}.txt" for stem in missing_stems]
    extra_labels = [labels_dir / f"{stem}.txt" for stem in extra_stems]
    empty_labels = [path for path in label_files if path.stat().st_size == 0]

    return SplitHealth(
        dataset_name=dataset_name,
        split=split,
        image_count=len(image_files),
        label_count=len(label_files),
        missing_labels=missing_labels,
        extra_labels=extra_labels,
        empty_labels=empty_labels,
        labels_expected=labels_expected,
    )


def run_dataset_check() -> dict[str, Path]:
    project_root = detect_project_root(Path(__file__).resolve().parent)
    compare_root = project_root / "runs" / "compare_yolov8_vs_yolo26"
    output_root = compare_root / "dataset_health"
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Processed_Data": project_root / "Processed_Data",
    }

    required_paths = [
        "Processed_Data/train/images",
        "Processed_Data/train/labels",
        "Processed_Data/valid/images",
        "Processed_Data/valid/labels",
        "Processed_Data/test/images",
        "Processed_Data/test/labels",
        "Processed_Data/data.yaml",
    ]

    required_rows: list[dict[str, str]] = []
    for rel_path in required_paths:
        absolute_path = project_root / rel_path
        required_rows.append(
            {
                "relative_path": rel_path,
                "absolute_path": str(absolute_path),
                "exists": str(absolute_path.exists()),
            }
        )

    split_rows: list[dict[str, str | int]] = []
    issue_rows: list[dict[str, str]] = []
    class_rows: list[dict[str, str]] = []

    for dataset_name, dataset_dir in datasets.items():
        dataset_yaml = dataset_dir / "data.yaml"
        nc = None
        names: list[str] = []
        if dataset_yaml.exists():
            nc, names = parse_class_info(dataset_yaml)

        class_rows.append(
            {
                "dataset": dataset_name,
                "data_yaml": str(dataset_yaml),
                "nc": "" if nc is None else str(nc),
                "names": ", ".join(names),
            }
        )

        for split, labels_expected in (("train", True), ("valid", True), ("test", False)):
            health = inspect_split(dataset_name, dataset_dir, split, labels_expected)
            split_rows.append(
                {
                    "dataset": health.dataset_name,
                    "split": health.split,
                    "image_count": health.image_count,
                    "label_count": health.label_count,
                    "missing_label_count": len(health.missing_labels),
                    "extra_label_count": len(health.extra_labels),
                    "empty_label_count": len(health.empty_labels),
                    "labels_expected": str(health.labels_expected),
                }
            )

            for path in health.missing_labels:
                issue_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "issue_type": "missing_label_file",
                        "file_path": str(path),
                    }
                )
            for path in health.extra_labels:
                issue_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "issue_type": "extra_label_file",
                        "file_path": str(path),
                    }
                )
            for path in health.empty_labels:
                issue_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "issue_type": "empty_label_file",
                        "file_path": str(path),
                    }
                )

    summary_csv = output_root / "dataset_health_summary.csv"
    issues_csv = output_root / "dataset_health_issues.csv"
    required_csv = output_root / "dataset_health_required_paths.csv"
    report_md = output_root / "dataset_health_report.md"

    write_csv(
        required_csv,
        required_rows,
        fieldnames=["relative_path", "absolute_path", "exists"],
    )
    write_csv(
        summary_csv,
        split_rows,
        fieldnames=[
            "dataset",
            "split",
            "image_count",
            "label_count",
            "missing_label_count",
            "extra_label_count",
            "empty_label_count",
            "labels_expected",
        ],
    )
    write_csv(
        issues_csv,
        issue_rows,
        fieldnames=["dataset", "split", "issue_type", "file_path"],
    )

    total_missing = sum(int(to_float(row["missing_label_count"]) or 0) for row in split_rows)
    total_extra = sum(int(to_float(row["extra_label_count"]) or 0) for row in split_rows)
    total_empty = sum(int(to_float(row["empty_label_count"]) or 0) for row in split_rows)

    lines: list[str] = []
    lines.append("# Dataset Health Report")
    lines.append("")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- Required paths report: `{required_csv}`")
    lines.append(f"- Split summary CSV: `{summary_csv}`")
    lines.append(f"- Issues CSV: `{issues_csv}`")
    lines.append("")
    lines.append("## Required Path Checks")
    lines.append("")
    lines.append("| Relative Path | Exists |")
    lines.append("|---|---|")
    for row in required_rows:
        lines.append(f"| `{row['relative_path']}` | {row['exists']} |")
    lines.append("")
    lines.append("## Class Information")
    lines.append("")
    lines.append("| Dataset | nc | names |")
    lines.append("|---|---:|---|")
    for row in class_rows:
        lines.append(f"| {row['dataset']} | {row['nc'] or 'n/a'} | {row['names'] or 'n/a'} |")
    lines.append("")
    lines.append("## Split Counts and Label Health")
    lines.append("")
    lines.append("| Dataset | Split | Images | Labels | Missing Labels | Extra Labels | Empty Labels | Labels Expected |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in split_rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['image_count']} | {row['label_count']} | "
            f"{row['missing_label_count']} | {row['extra_label_count']} | {row['empty_label_count']} | {row['labels_expected']} |"
        )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Total missing label files: **{total_missing}**")
    lines.append(f"- Total extra label files: **{total_extra}**")
    lines.append(f"- Total empty label files: **{total_empty}**")
    lines.append("")
    lines.append("Validation note: evaluation uses the `test` split.")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print("Dataset check completed.")
    print(f"Project root: {project_root}")
    print(f"Required paths CSV: {required_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Issues CSV: {issues_csv}")
    print(f"Markdown report: {report_md}")
    print(f"Total missing labels: {total_missing}")
    print(f"Total extra labels: {total_extra}")
    print(f"Total empty labels: {total_empty}")

    return {
        "required_csv": required_csv,
        "summary_csv": summary_csv,
        "issues_csv": issues_csv,
        "report_md": report_md,
    }


if __name__ == "__main__":
    run_dataset_check()

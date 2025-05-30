import json
import hashlib

from dataclasses import dataclass
from typing import List, Set, Optional
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from core.repositories.repo_files import FileItem
from evaluation.args import CMDArgs


class DatasetMetadata(BaseModel):
    questions_str_hash: str
    questions_split_hash: str
    files_list: Set[str]
    created_ts: datetime


@dataclass
class DatasetFiles:
    dataset_dir: Path
    metadata_file: Path
    questions_str_file: Path
    questions_split_file: Path

    @classmethod
    def new(cls, args: CMDArgs) -> 'DatasetFiles':
        return DatasetFiles(
            args.dataset_dir,
            args.dataset_dir.joinpath(".metadata.json"),
            args.dataset_dir.joinpath("questions_str.json"),
            args.dataset_dir.joinpath("questions_split.json")
        )


def verify_dataset_integrity_or_create_metadata(
        dataset_files: DatasetFiles,
        eval_files: List[FileItem],
):
    metadata = dataset_metadata_read(dataset_files.metadata_file)
    if metadata:
        questions_str_hash: str = file_md5(dataset_files.questions_str_file)
        questions_split_hash = file_md5(dataset_files.questions_split_file)

        try:
            assert metadata.questions_str_hash == questions_str_hash
            assert metadata.questions_split_hash == questions_split_hash
            assert metadata.files_list == {f.file_name_orig for f in eval_files}
        except Exception as _e:
            raise Exception(f"Dataset integrity is compromised:\nquestions_str or questions_split or files were changed.\n"
                            f"Dataset can no longer be used until changes are rolled back.")

    else:
        dataset_metadata_write(
            dataset_files,
            eval_files
        )


def dataset_metadata_write(
        dataset_files: DatasetFiles,
        eval_files: List[FileItem]
):
    questions_str_hash: str = file_md5(dataset_files.questions_str_file)
    questions_split_hash = file_md5(dataset_files.questions_split_file)

    metadata = DatasetMetadata(
        questions_str_hash=questions_str_hash,
        questions_split_hash=questions_split_hash,

        files_list={f.file_name_orig for f in eval_files},
        created_ts=datetime.now()
    )

    dataset_files.metadata_file.write_text(metadata.model_dump_json(indent=2))


def dataset_metadata_read(
        metadata_file: Path,
) -> Optional[DatasetMetadata]:
    try:
        return DatasetMetadata.model_validate(json.loads(metadata_file.read_text()))
    except Exception as _e:
        return


def file_md5(file: Path) -> str:
    return hashlib.md5(file.read_bytes()).hexdigest()

from typing import List, Any, Set

from openai import OpenAI

from core.logger import info, error
from openai_wrappers.api_files import file_delete
from openai_wrappers.api_vector_store import vector_store_file_delete



def cleanup_orphaned_files(
        client: OpenAI,
        vector_store: Any,
        vector_store_files: List[Any],
        expected_filenames: Set[str]
) -> None:
    """Delete files that exist in vector store but not in disk content."""
    for vs_file in vector_store_files:
        # Get the filename from the OpenAI file object
        # todo: use concurrency
        try:
            # Find the corresponding file in OpenAI files
            openai_file = client.files.retrieve(file_id=vs_file.id)
            filename = openai_file.filename

            if filename not in expected_filenames:
                info(f"Deleting orphaned file from vector store: {filename} (ID: {vs_file.id})")

                try:
                    # First delete from vector store
                    vector_store_file_delete(client, vector_store.id, vs_file.id)
                    info(f"Successfully removed file {filename} from vector store {vector_store.id}")

                    # Then delete the file itself
                    file_delete(client, vs_file.id)
                    info(f"Successfully deleted file {filename} (ID: {vs_file.id})")
                except Exception as e:
                    error(f"Error deleting orphaned file {filename} (ID: {vs_file.id}): {str(e)}")
        except Exception as e:
            error(f"Error retrieving file info for ID {vs_file.id}: {str(e)}")

# todo: implement

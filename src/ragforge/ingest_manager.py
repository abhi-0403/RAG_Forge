"""
ingestion_manager.py

Handles incremental document ingestion for RAG_Forge.

Responsibilities
----------------
1. Scan PDF directory
2. Compute SHA-256 hash
3. Detect new/modified PDFs
4. Maintain metadata.json
5. Coordinate loading, chunking, embedding and indexing
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict


class IngestionManager:
    """
    Handles incremental ingestion of PDF documents.
    """

    def __init__(
        self,
        pdf_directory: str = "data/raw_pdfs",
        metadata_file: str = "data/processed/metadata.json",
    ):
        """
        Initialize the ingestion manager.
        """

        self.pdf_directory = pdf_directory
        self.metadata_file = metadata_file

        # Ensure directories exist
        os.makedirs(self.pdf_directory, exist_ok=True)
        os.makedirs(
            os.path.dirname(self.metadata_file),
            exist_ok=True,
        )

        # Create metadata.json if missing
        if not os.path.exists(self.metadata_file):
            with open(
                self.metadata_file,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump({}, file, indent=4)

    # ==========================================================
    # Metadata
    # ==========================================================

    def load_metadata(self) -> Dict:
        """
        Load metadata.json
        """

        try:

            with open(
                self.metadata_file,
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except Exception:

            return {}

    def save_metadata(
        self,
        metadata: Dict,
    ):
        """
        Save metadata.json
        """

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

    # ==========================================================
    # Document Utilities
    # ==========================================================

    def generate_document_id(self) -> str:
        """
        Generate a unique document id.
        """

        return str(uuid.uuid4())

    def compute_file_hash(
        self,
        file_path: str,
    ) -> str:
        """
        Compute SHA-256 hash of a file.
        """

        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb",
        ) as file:

            while True:

                chunk = file.read(8192)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # ==========================================================
    # Document Status
    # ==========================================================

    def is_new_document(
        self,
        filename: str,
        metadata: Dict,
    ) -> bool:
        """
        Check whether document is new.
        """

        return filename not in metadata

    def has_document_changed(
        self,
        filename: str,
        file_hash: str,
        metadata: Dict,
    ) -> bool:
        """
        Check whether document hash changed.
        """

        if filename not in metadata:
            return False

        return metadata[filename]["sha256"] != file_hash

    # ==========================================================
    # PDF Scanning
    # ==========================================================

    def scan_pdf_directory(self):
        """
        Scan the raw PDF directory and return all PDF filenames.

        Returns:
            List of PDF filenames.
        """

        pdf_files = []

        for filename in os.listdir(self.pdf_directory):

            file_path = os.path.join(
                self.pdf_directory,
                filename,
            )

            if (
                os.path.isfile(file_path)
                and filename.lower().endswith(".pdf")
            ):
                pdf_files.append(filename)

        pdf_files.sort()

        return pdf_files

    # ==========================================================
    # Process Document
    # ==========================================================

    def process_document(
        self,
        filename: str,
        document_id: str,
        document_loader,
        text_splitter,
        embedding_manager,
        vector_store,
    ):
        """
        Process a single PDF document.

        Pipeline:
            Load PDF
                ↓
            Split into chunks
                ↓
            Generate embeddings
                ↓
            Add chunks to vector database

        Args:
            filename: Name of the PDF file.
            document_id: Unique ID assigned to the document.
            document_loader: DocumentLoader instance.
            text_splitter: TextSplitter instance.
            embedding_manager: EmbeddingManager instance.
            vector_store: VectorStore instance.

        Returns:
            Number of chunks indexed.
        """

        file_path = os.path.join(
            self.pdf_directory,
            filename,
        )

        print("\n" + "=" * 80)
        print(f"Processing Document : {filename}")
        print(f"Document ID        : {document_id}")
        print("=" * 80)

        # ------------------------------------------------------
        # Load PDF
        # ------------------------------------------------------

        documents = document_loader.load_pdf(
            file_path
        )

        if not documents:
            print(
                f"No documents were loaded from: {filename}"
            )
            return 0

        print(
            f"Pages Loaded       : {len(documents)}"
        )

        # ------------------------------------------------------
        # Split Document
        # ------------------------------------------------------

        chunks = text_splitter.split_documents(
            documents
        )

        if not chunks:
            print(
                f"No chunks generated for: {filename}"
            )
            return 0

        print(
            f"Chunks Generated   : {len(chunks)}"
        )

        # ------------------------------------------------------
        # Add Document Metadata to Chunks
        # ------------------------------------------------------

        for chunk in chunks:

            chunk.metadata["document_id"] = document_id
            chunk.metadata["source_file"] = filename

        # ------------------------------------------------------
        # Generate Embeddings
        # ------------------------------------------------------

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = (
            embedding_manager.generate_embeddings(
                texts
            )
        )

        if len(embeddings) == 0:
            print(
                f"No embeddings generated for: {filename}"
            )
            return 0

        # ------------------------------------------------------
        # Store in Vector Database
        # ------------------------------------------------------

        vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            document_id=document_id,
        )

        print(
            f"Successfully indexed: {filename}"
        )

        return len(chunks)

    # ==========================================================
    # Delete Existing Document
    # ==========================================================

    def remove_existing_document(
        self,
        document_id: str,
        vector_store,
    ):
        """
        Delete all existing chunks belonging to a document.

        Args:
            document_id: Unique ID of the existing document.
            vector_store: VectorStore instance.
        """

        print(
            f"\nRemoving old vectors for document: "
            f"{document_id}"
        )

        vector_store.delete_document(
            document_id
        )

    # ==========================================================
    # Main Ingestion Pipeline
    # ==========================================================

    def ingest(
        self,
        document_loader,
        text_splitter,
        embedding_manager,
        vector_store,
    ):
        """
        Incrementally ingest PDF documents.

        Workflow
        --------
        1. Scan PDF directory
        2. Read metadata.json
        3. Detect:
            - New PDFs
            - Modified PDFs
            - Unchanged PDFs
        4. Index only required PDFs
        5. Update metadata.json
        """

        metadata = self.load_metadata()

        pdf_files = self.scan_pdf_directory()

        if not pdf_files:

            print("\nNo PDF files found.")

            return

        print("\n" + "=" * 80)
        print("INCREMENTAL INGESTION")
        print("=" * 80)

        print(f"Found {len(pdf_files)} PDF(s)\n")

        new_documents = 0
        updated_documents = 0
        skipped_documents = 0

        total_chunks = 0

        for filename in pdf_files:

            file_path = os.path.join(
                self.pdf_directory,
                filename,
            )

            file_hash = self.compute_file_hash(
                file_path
            )

            # --------------------------------------------------
            # NEW DOCUMENT
            # --------------------------------------------------

            if self.is_new_document(
                filename,
                metadata,
            ):

                print(f"\n[NEW] {filename}")

                document_id = self.generate_document_id()

                chunk_count = self.process_document(
                    filename=filename,
                    document_id=document_id,
                    document_loader=document_loader,
                    text_splitter=text_splitter,
                    embedding_manager=embedding_manager,
                    vector_store=vector_store,
                )

                metadata[filename] = {
                    "document_id": document_id,
                    "sha256": file_hash,
                    "chunks": chunk_count,
                    "indexed_at": datetime.now().isoformat(),
                }

                new_documents += 1
                total_chunks += chunk_count

            # --------------------------------------------------
            # MODIFIED DOCUMENT
            # --------------------------------------------------

            elif self.has_document_changed(
                filename,
                file_hash,
                metadata,
            ):

                print(f"\n[UPDATED] {filename}")

                old_document_id = metadata[
                    filename
                ]["document_id"]

                self.remove_existing_document(
                    old_document_id,
                    vector_store,
                )

                new_document_id = self.generate_document_id()

                chunk_count = self.process_document(
                    filename=filename,
                    document_id=new_document_id,
                    document_loader=document_loader,
                    text_splitter=text_splitter,
                    embedding_manager=embedding_manager,
                    vector_store=vector_store,
                )

                metadata[filename] = {
                    "document_id": new_document_id,
                    "sha256": file_hash,
                    "chunks": chunk_count,
                    "indexed_at": datetime.now().isoformat(),
                }

                updated_documents += 1
                total_chunks += chunk_count

            # --------------------------------------------------
            # UNCHANGED DOCUMENT
            # --------------------------------------------------

            else:

                print(f"\n[SKIPPED] {filename}")

                skipped_documents += 1

        # ------------------------------------------------------
        # Save metadata
        # ------------------------------------------------------

        self.save_metadata(
            metadata
        )

        print("\n" + "=" * 80)
        print("INGESTION SUMMARY")
        print("=" * 80)

        print(f"New Documents      : {new_documents}")
        print(f"Updated Documents  : {updated_documents}")
        print(f"Skipped Documents  : {skipped_documents}")
        print(f"Chunks Indexed     : {total_chunks}")
        print(
            f"Vector Count       : "
            f"{vector_store.get_collection_count()}"
        )

        print("\nMetadata updated successfully.")
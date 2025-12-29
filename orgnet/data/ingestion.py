"""Data ingestion from various sources."""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from orgnet.data.models import (
    Person,
    Interaction,
    Document,
    Meeting,
    CodeCommit,
    HRISRecord,
    InteractionType,
)
from orgnet.config import Config
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class DataIngester:
    """Ingests data from various organizational sources."""

    def __init__(self, config: Config):
        """
        Initialize data ingester.

        Args:
            config: Configuration object
        """
        self.config = config
        self.retention_days = self._get_retention_days()

    def _get_retention_days(self) -> int:
        """Get data retention period in days."""
        # Get the minimum retention across all enabled sources
        retention_days = []
        for source in ["email", "slack", "teams", "calendar", "documents", "code"]:
            if self.config.is_data_source_enabled(source):
                source_config = self.config.get_data_source_config(source)
                retention_days.append(source_config.get("retention_days", 90))

        return min(retention_days) if retention_days else 90

    def ingest_email(
        self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> List[Interaction]:
        """
        Ingest email data.

        Args:
            data_path: Path to CSV file with email data
            data: DataFrame with email data (columns: sender, recipient, timestamp, etc.)

        Returns:
            List of Interaction objects
        """
        if not self.config.is_data_source_enabled("email"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            data = pd.read_csv(data_path)

        interactions = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for _, row in data.iterrows():
            timestamp = pd.to_datetime(row["timestamp"])
            if timestamp < cutoff_date:
                continue

            # Skip rows with missing source or target
            sender_id = row["sender_id"]
            recipient_id = row["recipient_id"]
            if pd.isna(sender_id) or pd.isna(recipient_id):
                continue

            interaction = Interaction(
                id=f"email_{row.get('id', len(interactions))}",
                source_id=str(sender_id),
                target_id=str(recipient_id),
                interaction_type=InteractionType.EMAIL,
                timestamp=timestamp,
                channel=row.get("subject", None),
                response_time_seconds=row.get("response_time_seconds") if not pd.isna(row.get("response_time_seconds")) else None,
                is_reciprocal=row.get("is_reciprocal", False),
                metadata={
                    "cc": row.get("cc", []),
                    "bcc": row.get("bcc", []),
                    "has_attachment": row.get("has_attachment", False),
                },
            )
            interactions.append(interaction)

        return interactions

    def ingest_slack(
        self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> List[Interaction]:
        """
        Ingest Slack data.

        Args:
            data_path: Path to CSV file with Slack data
            data: DataFrame with Slack message data

        Returns:
            List of Interaction objects
        """
        if not self.config.is_data_source_enabled("slack"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            data = pd.read_csv(data_path)

        interactions = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for _, row in data.iterrows():
            timestamp = pd.to_datetime(row["timestamp"])
            if timestamp < cutoff_date:
                continue

            # Skip rows with missing user_id
            user_id = row["user_id"]
            if pd.isna(user_id):
                continue

            # For channel messages, skip (no person-to-person interaction)
            # Only process DMs (direct messages between people)
            target_user_id = row.get("target_user_id")
            if pd.isna(target_user_id) or not row.get("is_dm", False):
                continue  # Skip channel messages for now

            interaction = Interaction(
                id=f"slack_{row.get('id', len(interactions))}",
                source_id=str(user_id),
                target_id=str(target_user_id),
                interaction_type=InteractionType.SLACK,
                timestamp=timestamp,
                channel=row.get("channel", None),
                thread_id=row.get("thread_ts", None),
                response_time_seconds=row.get("response_time_seconds") if not pd.isna(row.get("response_time_seconds")) else None,
                metadata={
                    "is_dm": row.get("is_dm", False),
                    "has_reaction": row.get("has_reaction", False),
                    "mentions": row.get("mentions", []),
                },
            )
            interactions.append(interaction)

        return interactions

    def ingest_calendar(
        self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> List[Meeting]:
        """
        Ingest calendar/meeting data.

        Args:
            data_path: Path to CSV file with meeting data
            data: DataFrame with meeting data

        Returns:
            List of Meeting objects
        """
        if not self.config.is_data_source_enabled("calendar"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            data = pd.read_csv(data_path)

        meetings = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for _, row in data.iterrows():
            start_time = pd.to_datetime(row["start_time"])
            if start_time < cutoff_date:
                continue

            end_time = pd.to_datetime(row["end_time"])
            duration = (end_time - start_time).total_seconds() / 60

            attendee_ids_raw = row.get("attendee_ids", "")
            if pd.isna(attendee_ids_raw):
                attendee_ids = []
            elif isinstance(attendee_ids_raw, str):
                attendee_ids = [aid.strip() for aid in attendee_ids_raw.split(",") if aid.strip()]
            else:
                attendee_ids = list(attendee_ids_raw) if attendee_ids_raw else []

            meeting = Meeting(
                id=f"meeting_{row.get('id', len(meetings))}",
                organizer_id=str(row["organizer_id"]),
                attendee_ids=attendee_ids,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=duration,
                is_recurring=row.get("is_recurring", False),
                meeting_type=row.get("meeting_type", None),
                metadata={
                    "subject": row.get("subject", None),
                    "location": row.get("location", None),
                },
            )
            meetings.append(meeting)

        return meetings

    def ingest_documents(
        self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> List[Document]:
        """
        Ingest document collaboration data.

        Args:
            data_path: Path to CSV file with document data
            data: DataFrame with document data

        Returns:
            List of Document objects
        """
        if not self.config.is_data_source_enabled("documents"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            data = pd.read_csv(data_path)

        documents = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for _, row in data.iterrows():
            created_at = pd.to_datetime(row["created_at"])
            if created_at < cutoff_date:
                continue

            author_ids_raw = row["author_ids"]
            if pd.isna(author_ids_raw):
                author_ids = []
            elif isinstance(author_ids_raw, str):
                author_ids = [aid.strip() for aid in author_ids_raw.split(",") if aid.strip()]
            else:
                author_ids = list(author_ids_raw) if author_ids_raw else []

            editor_ids_raw = row.get("editor_ids", "")
            if pd.isna(editor_ids_raw) or editor_ids_raw == "":
                editor_ids = []
            elif isinstance(editor_ids_raw, str):
                editor_ids = [eid.strip() for eid in editor_ids_raw.split(",") if eid.strip()]
            else:
                editor_ids = list(editor_ids_raw) if editor_ids_raw else []

            document = Document(
                id=f"doc_{row.get('id', len(documents))}",
                title=row.get("title", "Untitled"),
                author_ids=author_ids,
                editor_ids=editor_ids,
                created_at=created_at,
                last_modified=pd.to_datetime(row["last_modified"]),
                document_type=row.get("document_type", None),
                platform=row.get("platform", None),
                metadata={"url": row.get("url", None), "size": row.get("size", None)},
            )
            documents.append(document)

        return documents

    def ingest_code(
        self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> List[CodeCommit]:
        """
        Ingest code repository data.

        Args:
            data_path: Path to CSV file with commit data
            data: DataFrame with commit data

        Returns:
            List of CodeCommit objects
        """
        if not self.config.is_data_source_enabled("code"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            data = pd.read_csv(data_path)

        commits = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for _, row in data.iterrows():
            timestamp = pd.to_datetime(row["timestamp"])
            if timestamp < cutoff_date:
                continue

            file_paths_raw = row["file_paths"]
            if pd.isna(file_paths_raw):
                file_paths = []
            elif isinstance(file_paths_raw, str):
                file_paths = [fp.strip() for fp in file_paths_raw.split(",") if fp.strip()]
            else:
                file_paths = list(file_paths_raw) if file_paths_raw else []

            reviewer_ids_raw = row.get("reviewer_ids", "")
            if pd.isna(reviewer_ids_raw) or reviewer_ids_raw == "":
                reviewer_ids = []
            elif isinstance(reviewer_ids_raw, str):
                reviewer_ids = [rid.strip() for rid in reviewer_ids_raw.split(",") if rid.strip()]
            else:
                reviewer_ids = list(reviewer_ids_raw) if reviewer_ids_raw else []

            commit = CodeCommit(
                id=f"commit_{row.get('id', len(commits))}",
                author_id=row["author_id"],
                repository=row["repository"],
                file_paths=file_paths,
                timestamp=timestamp,
                reviewer_ids=reviewer_ids,
                is_merge=row.get("is_merge", False),
                metadata={
                    "commit_hash": row.get("commit_hash", None),
                    "message": row.get("message", None),
                },
            )
            commits.append(commit)

        return commits

    def ingest_hris(
        self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ) -> List[HRISRecord]:
        """
        Ingest HRIS data.

        Args:
            data_path: Path to CSV file with HRIS data
            data: DataFrame with HRIS data

        Returns:
            List of HRISRecord objects
        """
        if not self.config.is_data_source_enabled("hris"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            data = pd.read_csv(data_path)

        records = []

        for _, row in data.iterrows():
            record = HRISRecord(
                person_id=row["person_id"],
                department=row["department"],
                role=row["role"],
                manager_id=row.get("manager_id", None),
                team=row["team"],
                start_date=pd.to_datetime(row["start_date"]),
                location=row.get("location", None),
                job_level=row.get("job_level", None),
                metadata={
                    "employee_id": row.get("employee_id", None),
                    "status": row.get("status", "active"),
                },
            )
            records.append(record)

        return records

    def create_people_from_hris(self, hris_records: List[HRISRecord]) -> List[Person]:
        """
        Create Person objects from HRIS records.

        Args:
            hris_records: List of HRISRecord objects

        Returns:
            List of Person objects
        """
        people = []
        seen_ids = set()

        for record in hris_records:
            if record.person_id in seen_ids:
                continue
            seen_ids.add(record.person_id)

            # Calculate tenure
            tenure_days = (datetime.now() - record.start_date).days if record.start_date else None

            person = Person(
                id=record.person_id,
                name=record.metadata.get("name", record.person_id),
                email=record.metadata.get("email", f"{record.person_id}@company.com"),
                department=record.department,
                role=record.role,
                manager_id=record.manager_id,
                team=record.team,
                location=record.location,
                job_level=record.job_level,
                tenure_days=tenure_days,
                metadata=record.metadata,
            )
            people.append(person)

        return people

"""Data ingestion from various sources."""

import pandas as pd
import email
from email.utils import parsedate_to_datetime, parseaddr
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import random

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

    def _detect_email_format(self, data_path: str) -> str:
        """
        Detect email data format (CSV or maildir).

        Args:
            data_path: Path to email data

        Returns:
            Format string: 'csv' or 'maildir'
        """
        path = Path(data_path)

        # Check if it's a directory (maildir) or file (CSV)
        if path.is_dir():
            # Check for maildir structure (has cur/, new/, tmp/ subdirectories)
            if any((path / subdir).exists() for subdir in ["cur", "new", "tmp"]):
                return "maildir"
            # Otherwise assume it's a maildir with user folders
            return "maildir"
        elif path.is_file() and path.suffix.lower() == ".csv":
            return "csv"
        else:
            # Default to CSV for backward compatibility
            logger.warning(f"Could not detect format for {data_path}, defaulting to CSV")
            return "csv"

    def _parse_maildir_email(self, email_path: Path, email_id: str) -> Optional[Dict]:
        """
        Parse a single email file from maildir format.

        Args:
            email_path: Path to email file
            email_id: Unique identifier for this email

        Returns:
            Dictionary with email data or None if parsing fails
        """
        try:
            with open(email_path, "rb") as f:
                msg = email.message_from_bytes(f.read())

            # Extract headers
            from_addr = parseaddr(msg.get("From", ""))[1]
            to_addrs = [parseaddr(addr)[1] for addr in msg.get_all("To", [])]
            cc_addrs = [parseaddr(addr)[1] for addr in msg.get_all("Cc", [])]
            bcc_addrs = [parseaddr(addr)[1] for addr in msg.get_all("Bcc", [])]

            # Get date
            date_str = msg.get("Date")
            if date_str:
                try:
                    timestamp = parsedate_to_datetime(date_str)
                except (ValueError, TypeError):
                    timestamp = datetime.fromtimestamp(email_path.stat().st_mtime)
            else:
                timestamp = datetime.fromtimestamp(email_path.stat().st_mtime)

            # Get subject
            subject = msg.get("Subject", "")

            # Get message ID for threading
            message_id = msg.get("Message-ID", "")
            in_reply_to = msg.get("In-Reply-To", "")
            references = msg.get("References", "")

            # Extract body (text/plain preferred, fallback to text/html)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                        except Exception:
                            pass
                    elif content_type == "text/html" and not body:
                        try:
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        except Exception:
                            pass
            else:
                try:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                except Exception:
                    pass

            # Check for attachments
            has_attachment = any(
                part.get_content_disposition() == "attachment"
                for part in msg.walk()
                if msg.is_multipart()
            )

            return {
                "id": email_id,
                "from": from_addr,
                "to": to_addrs,
                "cc": cc_addrs,
                "bcc": bcc_addrs,
                "timestamp": timestamp,
                "subject": subject,
                "body": body,
                "message_id": message_id,
                "in_reply_to": in_reply_to,
                "references": references,
                "has_attachment": has_attachment,
            }
        except Exception as e:
            logger.warning(f"Failed to parse email {email_path}: {e}")
            return None

    def _load_maildir(
        self,
        maildir_path: str,
        max_rows: Optional[int] = None,
        sample_size: Optional[int] = None,
        user_filter: Optional[List[str]] = None,
        folder_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load emails from maildir format (Enron-style).

        Args:
            maildir_path: Path to maildir directory
            max_rows: Maximum number of emails to load
            sample_size: Random sample size (if None, loads all)
            user_filter: Optional list of user folders to include
            folder_filter: Optional list of folder types to include (e.g., ['sent', 'inbox'])

        Returns:
            DataFrame with email data
        """
        maildir = Path(maildir_path)
        if not maildir.exists():
            raise ValueError(f"Maildir path does not exist: {maildir_path}")

        logger.info(f"Loading emails from maildir: {maildir_path}")

        emails = []
        email_count = 0

        # Enron maildir structure: maildir/user-folder/cur/ or maildir/user-folder/new/
        # Also supports: maildir/cur/ and maildir/new/ (single user)

        # Find all email files
        email_files = []

        if (maildir / "cur").exists() or (maildir / "new").exists():
            # Single user maildir
            for subdir in ["cur", "new"]:
                subdir_path = maildir / subdir
                if subdir_path.exists():
                    email_files.extend(subdir_path.glob("*"))
        else:
            # Multi-user maildir (Enron style)
            for user_folder in maildir.iterdir():
                if not user_folder.is_dir():
                    continue

                if user_filter and user_folder.name not in user_filter:
                    continue

                # Check subdirectories (cur, new, or direct folders like 'sent', 'inbox')
                for subdir in ["cur", "new"]:
                    subdir_path = user_folder / subdir
                    if subdir_path.exists():
                        email_files.extend(subdir_path.glob("*"))

                # Also check for direct folder structure (sent, inbox, etc.)
                for folder in user_folder.iterdir():
                    if folder.is_dir() and folder.name not in ["cur", "new", "tmp"]:
                        if folder_filter and folder.name not in folder_filter:
                            continue
                        for subdir in ["cur", "new"]:
                            subdir_path = folder / subdir
                            if subdir_path.exists():
                                email_files.extend(subdir_path.glob("*"))

        # Filter out directories
        email_files = [f for f in email_files if f.is_file()]

        logger.info(f"Found {len(email_files)} email files")

        # Apply sampling if requested
        if sample_size and len(email_files) > sample_size:
            email_files = random.sample(email_files, sample_size)
            logger.info(f"Sampled {sample_size} emails")

        # Apply max_rows limit
        if max_rows:
            email_files = email_files[:max_rows]

        from orgnet.utils.performance import parallel_map

        def parse_email_wrapper(email_file):
            email_id = f"email_{len(emails)}"
            return self._parse_maildir_email(email_file, email_id)

        if len(email_files) > 100:
            parsed_results = parallel_map(parse_email_wrapper, email_files, n_jobs=-1)
            emails.extend([p for p in parsed_results if p is not None])
            email_count = len(emails)
        else:
            for email_file in email_files:
                email_id = f"email_{email_count}"
                parsed = self._parse_maildir_email(email_file, email_id)
                if parsed:
                    emails.append(parsed)
                    email_count += 1

                if max_rows and email_count >= max_rows:
                    break

        if not emails:
            logger.warning("No emails parsed from maildir")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(emails)

        # Expand 'to' list into multiple rows (one per recipient)
        rows = []
        for _, email_row in df.iterrows():
            from_addr = email_row["from"]
            to_addrs = email_row["to"] if isinstance(email_row["to"], list) else []

            if not to_addrs:
                # No recipients, skip
                continue

            for to_addr in to_addrs:
                rows.append(
                    {
                        "id": email_row["id"],
                        "sender": from_addr,
                        "recipient": to_addr,
                        "timestamp": email_row["timestamp"],
                        "subject": email_row["subject"],
                        "cc": email_row.get("cc", []),
                        "bcc": email_row.get("bcc", []),
                        "has_attachment": email_row.get("has_attachment", False),
                        "message_id": email_row.get("message_id", ""),
                        "in_reply_to": email_row.get("in_reply_to", ""),
                        "references": email_row.get("references", ""),
                        "body": email_row.get("body", ""),
                    }
                )

        df_expanded = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df_expanded)} email interactions from {len(emails)} emails")

        return df_expanded

    def ingest_email(
        self,
        data_path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        data_format: Optional[str] = None,
        max_rows: Optional[int] = None,
        sample_size: Optional[int] = None,
        user_filter: Optional[List[str]] = None,
        folder_filter: Optional[List[str]] = None,
        email_to_person_map: Optional[Dict[str, str]] = None,
    ) -> List[Interaction]:
        """
        Ingest email data from CSV or maildir format.

        Args:
            data_path: Path to CSV file or maildir directory with email data
            data: DataFrame with email data (columns: sender, recipient, timestamp, etc.)
            data_format: Format type ('csv', 'maildir', or 'auto' for auto-detection)
            max_rows: Maximum number of emails to load (maildir only)
            sample_size: Random sample size (maildir only)
            user_filter: Optional list of user folders to include (maildir only)
            folder_filter: Optional list of folder types to include (maildir only)
            email_to_person_map: Optional mapping from email addresses to person IDs

        Returns:
            List of Interaction objects
        """
        if not self.config.is_data_source_enabled("email"):
            return []

        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")

            # Detect format if not specified
            if data_format is None or data_format == "auto":
                data_format = self._detect_email_format(data_path)

            # Load data based on format
            if data_format == "maildir":
                data = self._load_maildir(
                    data_path,
                    max_rows=max_rows,
                    sample_size=sample_size,
                    user_filter=user_filter,
                    folder_filter=folder_filter,
                )

                # Map email addresses to person IDs if mapping provided
                if email_to_person_map:
                    data["sender_id"] = data["sender"].map(email_to_person_map)
                    data["recipient_id"] = data["recipient"].map(email_to_person_map)
                else:
                    # Use email addresses as IDs if no mapping provided
                    data["sender_id"] = data["sender"]
                    data["recipient_id"] = data["recipient"]
            else:
                # CSV format
                data = pd.read_csv(data_path)

        interactions = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for _, row in data.iterrows():
            timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
            if pd.isna(timestamp) or timestamp < cutoff_date:
                continue

            # Get sender and recipient IDs
            # Support both 'sender_id'/'recipient_id' (CSV) and 'sender'/'recipient' (maildir)
            sender_id = row.get("sender_id") or row.get("sender")
            recipient_id = row.get("recipient_id") or row.get("recipient")

            if pd.isna(sender_id) or pd.isna(recipient_id) or not sender_id or not recipient_id:
                continue

            # Extract CC/BCC lists
            cc_list = row.get("cc", [])
            if isinstance(cc_list, str):
                cc_list = [c.strip() for c in cc_list.split(",") if c.strip()]
            elif not isinstance(cc_list, list):
                cc_list = []

            bcc_list = row.get("bcc", [])
            if isinstance(bcc_list, str):
                bcc_list = [c.strip() for c in bcc_list.split(",") if c.strip()]
            elif not isinstance(bcc_list, list):
                bcc_list = []

            interaction = Interaction(
                id=str(row.get("id", f"email_{len(interactions)}")),
                source_id=str(sender_id),
                target_id=str(recipient_id),
                interaction_type=InteractionType.EMAIL,
                timestamp=timestamp,
                channel=row.get("subject", None),
                response_time_seconds=(
                    row.get("response_time_seconds")
                    if not pd.isna(row.get("response_time_seconds"))
                    else None
                ),
                is_reciprocal=row.get("is_reciprocal", False),
                content=row.get("body", None),  # Store email body if available
                metadata={
                    "cc": cc_list,
                    "bcc": bcc_list,
                    "has_attachment": row.get("has_attachment", False),
                    "message_id": row.get("message_id", ""),
                    "in_reply_to": row.get("in_reply_to", ""),
                    "references": row.get("references", ""),
                },
            )
            interactions.append(interaction)

        logger.info(f"Ingested {len(interactions)} email interactions")
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
                response_time_seconds=(
                    row.get("response_time_seconds")
                    if not pd.isna(row.get("response_time_seconds"))
                    else None
                ),
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

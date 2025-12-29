# Data Format Guide

This document describes the expected format for input data files.

## HRIS Data (Required)

File: `hris.csv`

Required columns:
- `person_id`: Unique identifier for each person
- `department`: Department name
- `role`: Job role/title
- `team`: Team name
- `start_date`: Start date (YYYY-MM-DD format)
- `manager_id`: Manager's person_id (optional)

Optional columns:
- `name`: Person's name
- `email`: Email address
- `location`: Physical location
- `job_level`: Job level (e.g., "Senior", "Principal")
- `employee_id`: Employee ID
- `status`: Employment status (e.g., "active")

Example:
```csv
person_id,department,role,team,start_date,manager_id,name,email
emp001,Engineering,Software Engineer,Backend,2020-01-15,emp010,John Doe,john@company.com
emp002,Product,Product Manager,Core Product,2019-06-01,emp011,Jane Smith,jane@company.com
```

## Email Data

File: `email.csv`

Required columns:
- `sender_id`: Person ID of sender
- `recipient_id`: Person ID of recipient
- `timestamp`: Email timestamp (ISO format)

Optional columns:
- `id`: Unique email ID
- `subject`: Email subject
- `response_time_seconds`: Response time in seconds
- `is_reciprocal`: Boolean indicating if conversation is reciprocal
- `cc`: Comma-separated list of CC'd person IDs
- `bcc`: Comma-separated list of BCC'd person IDs
- `has_attachment`: Boolean

Example:
```csv
id,sender_id,recipient_id,timestamp,subject,response_time_seconds,is_reciprocal
email001,emp001,emp002,2024-01-15T10:30:00,Project Update,3600,True
email002,emp002,emp001,2024-01-15T11:30:00,Re: Project Update,0,True
```

## Slack Data

File: `slack.csv`

Required columns:
- `user_id`: Person ID of message sender
- `timestamp`: Message timestamp (ISO format)

Optional columns:
- `id`: Unique message ID
- `target_user_id`: Person ID for direct messages
- `channel`: Channel name
- `thread_ts`: Thread timestamp (for threaded messages)
- `response_time_seconds`: Response time in seconds
- `is_dm`: Boolean indicating if it's a direct message
- `has_reaction`: Boolean
- `mentions`: Comma-separated list of mentioned person IDs

Example:
```csv
id,user_id,target_user_id,timestamp,channel,is_dm
slack001,emp001,emp002,2024-01-15T14:00:00,general,False
slack002,emp002,emp001,2024-01-15T14:05:00,,True
```

## Calendar/Meeting Data

File: `calendar.csv`

Required columns:
- `organizer_id`: Person ID of meeting organizer
- `start_time`: Meeting start time (ISO format)
- `end_time`: Meeting end time (ISO format)

Optional columns:
- `id`: Unique meeting ID
- `attendee_ids`: Comma-separated list of attendee person IDs
- `is_recurring`: Boolean
- `meeting_type`: Type of meeting (e.g., "1-on-1", "standup")
- `subject`: Meeting subject
- `location`: Meeting location

Example:
```csv
id,organizer_id,attendee_ids,start_time,end_time,is_recurring,meeting_type
meeting001,emp010,emp001,emp002,emp003,2024-01-15T09:00:00,2024-01-15T10:00:00,False,standup
meeting002,emp011,emp004,emp005,2024-01-15T14:00:00,2024-01-15T15:00:00,True,1-on-1
```

## Document Collaboration Data

File: `documents.csv`

Required columns:
- `author_ids`: Comma-separated list of author person IDs
- `created_at`: Document creation timestamp (ISO format)
- `last_modified`: Last modification timestamp (ISO format)

Optional columns:
- `id`: Unique document ID
- `title`: Document title
- `editor_ids`: Comma-separated list of editor person IDs
- `document_type`: Type of document (e.g., "wiki", "doc", "presentation")
- `platform`: Platform name (e.g., "sharepoint", "google_drive")
- `url`: Document URL
- `size`: Document size in bytes

Example:
```csv
id,title,author_ids,editor_ids,created_at,last_modified,platform
doc001,Project Plan,emp001,emp002,emp003,2024-01-01T00:00:00,2024-01-10T00:00:00,sharepoint
doc002,Design Spec,emp004,emp005,2024-01-05T00:00:00,2024-01-12T00:00:00,google_drive
```

## Code Repository Data

File: `code.csv`

Required columns:
- `author_id`: Person ID of commit author
- `repository`: Repository name
- `file_paths`: Comma-separated list of file paths changed
- `timestamp`: Commit timestamp (ISO format)

Optional columns:
- `id`: Unique commit ID
- `reviewer_ids`: Comma-separated list of reviewer person IDs
- `is_merge`: Boolean indicating if it's a merge commit
- `commit_hash`: Git commit hash
- `message`: Commit message

Example:
```csv
id,author_id,repository,file_paths,timestamp,reviewer_ids,is_merge
commit001,emp001,backend-api,src/api.py,src/models.py,2024-01-15T16:00:00,emp002,False
commit002,emp002,frontend-app,components/Button.tsx,2024-01-15T17:00:00,emp003,False
```

## Notes

1. **Person IDs**: All person IDs must match the `person_id` values in the HRIS data.

2. **Timestamps**: Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS) or any format that pandas can parse.

3. **Missing Data**: Optional columns can be omitted or left empty. The system will handle missing values gracefully.

4. **Data Privacy**: Ensure all data is properly anonymized and complies with privacy regulations before analysis.

5. **File Encoding**: Use UTF-8 encoding for all CSV files.

6. **Data Volume**: The system can handle large datasets, but processing time will increase with data size.


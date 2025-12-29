"""
Generate sample organizational network data using Faker.

This script creates realistic sample data files that match the DATA_FORMAT.md specification.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from pathlib import Path

# Initialize Faker
fake = Faker()
Faker.seed(42)  # For reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
NUM_PEOPLE = 50
NUM_DAYS = 90  # Generate data for last 90 days
OUTPUT_DIR = Path(__file__).parent / "data"

# Departments and roles
DEPARTMENTS = ["Engineering", "Product", "Design", "Sales", "Marketing", "Operations", "HR"]
ROLES = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Engineering Manager", "Tech Lead"],
    "Product": ["Product Manager", "Senior PM", "Product Director"],
    "Design": ["UX Designer", "UI Designer", "Design Lead"],
    "Sales": ["Sales Rep", "Sales Manager", "Account Executive"],
    "Marketing": ["Marketing Manager", "Content Manager", "Growth Marketer"],
    "Operations": ["Operations Manager", "Analyst", "Coordinator"],
    "HR": ["HR Manager", "Recruiter", "People Ops"],
}

TEAMS = {
    "Engineering": ["Backend", "Frontend", "Infrastructure", "Mobile"],
    "Product": ["Core Product", "Growth", "Platform"],
    "Design": ["Product Design", "Brand Design"],
    "Sales": ["Enterprise", "SMB", "Inside Sales"],
    "Marketing": ["Content", "Growth", "Brand"],
    "Operations": ["Business Ops", "Data Ops"],
    "HR": ["Talent", "People Ops"],
}

JOB_LEVELS = ["Individual Contributor", "Senior", "Principal", "Manager", "Director", "VP"]


def generate_hris_data(num_people: int) -> pd.DataFrame:
    """Generate HRIS data."""
    print(f"Generating HRIS data for {num_people} people...")
    
    people = []
    person_ids = [f"emp{i:03d}" for i in range(1, num_people + 1)]
    
    # Create some managers first
    managers = person_ids[:num_people // 5]
    
    for i, person_id in enumerate(person_ids):
        dept = random.choice(DEPARTMENTS)
        role = random.choice(ROLES[dept])
        team = random.choice(TEAMS[dept])
        
        # Start dates spread over last 2 years
        start_date = fake.date_between(start_date="-2y", end_date="today")
        
        # Calculate tenure
        tenure_days = (datetime.now().date() - start_date).days
        
        # Assign manager (some people have managers, some don't)
        manager_id = random.choice(managers) if i > len(managers) and random.random() > 0.3 else None
        
        # Job level based on role
        if "Manager" in role or "Director" in role or "VP" in role:
            job_level = random.choice(["Manager", "Director", "VP"])
        elif "Senior" in role or "Lead" in role:
            job_level = random.choice(["Senior", "Principal"])
        else:
            job_level = "Individual Contributor"
        
        people.append({
            "person_id": person_id,
            "name": fake.name(),
            "email": fake.email(),
            "department": dept,
            "role": role,
            "team": team,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "manager_id": manager_id,
            "location": fake.city(),
            "job_level": job_level,
            "employee_id": f"E{person_id}",
            "status": "active",
        })
    
    df = pd.DataFrame(people)
    return df


def generate_email_data(person_ids: list, num_days: int) -> pd.DataFrame:
    """Generate email interaction data."""
    print(f"Generating email data for {num_days} days...")
    
    emails = []
    num_emails = len(person_ids) * 5  # ~5 emails per person
    
    for i in range(num_emails):
        sender = random.choice(person_ids)
        recipient = random.choice([p for p in person_ids if p != sender])
        
        # Timestamp within last num_days
        timestamp = fake.date_time_between(
            start_date=f"-{num_days}d", end_date="now"
        )
        
        # Some emails have responses
        response_time = random.randint(300, 7200) if random.random() > 0.5 else None
        is_reciprocal = random.random() > 0.6
        
        emails.append({
            "id": f"email{i:05d}",
            "sender_id": sender,
            "recipient_id": recipient,
            "timestamp": timestamp.isoformat(),
            "subject": fake.sentence(nb_words=4),
            "response_time_seconds": response_time,
            "is_reciprocal": is_reciprocal,
            "has_attachment": random.random() > 0.7,
        })
    
    df = pd.DataFrame(emails)
    df = df.sort_values("timestamp")
    return df


def generate_slack_data(person_ids: list, num_days: int) -> pd.DataFrame:
    """Generate Slack message data."""
    print(f"Generating Slack data for {num_days} days...")
    
    messages = []
    num_messages = len(person_ids) * 8  # ~8 messages per person
    
    channels = ["general", "engineering", "product", "random", "announcements"]
    
    for i in range(num_messages):
        user = random.choice(person_ids)
        
        # 30% are DMs
        is_dm = random.random() < 0.3
        if is_dm:
            target_user = random.choice([p for p in person_ids if p != user])
            channel = None
        else:
            target_user = None
            channel = random.choice(channels)
        
        timestamp = fake.date_time_between(
            start_date=f"-{num_days}d", end_date="now"
        )
        
        messages.append({
            "id": f"slack{i:05d}",
            "user_id": user,
            "target_user_id": target_user,
            "timestamp": timestamp.isoformat(),
            "channel": channel,
            "is_dm": is_dm,
            "has_reaction": random.random() > 0.8,
            "mentions": ",".join(random.sample(person_ids, random.randint(0, 2))) if random.random() > 0.7 else "",
        })
    
    df = pd.DataFrame(messages)
    df = df.sort_values("timestamp")
    return df


def generate_calendar_data(person_ids: list, num_days: int) -> pd.DataFrame:
    """Generate calendar/meeting data."""
    print(f"Generating calendar data for {num_days} days...")
    
    meetings = []
    num_meetings = len(person_ids) * 2  # ~2 meetings per person
    
    meeting_types = ["standup", "1-on-1", "team_meeting", "all_hands", "planning", "retro"]
    
    for i in range(num_meetings):
        organizer = random.choice(person_ids)
        
        # Meeting duration: 30 min to 2 hours
        start_time = fake.date_time_between(
            start_date=f"-{num_days}d", end_date="now"
        )
        duration_minutes = random.choice([30, 60, 90, 120])
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Attendees (2-8 people)
        num_attendees = random.randint(2, min(8, len(person_ids)))
        attendees = random.sample(person_ids, num_attendees)
        if organizer not in attendees:
            attendees.append(organizer)
        
        meetings.append({
            "id": f"meeting{i:05d}",
            "organizer_id": organizer,
            "attendee_ids": ",".join(attendees),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "is_recurring": random.random() > 0.7,
            "meeting_type": random.choice(meeting_types),
            "subject": fake.sentence(nb_words=3),
            "location": random.choice(["Zoom", "Conference Room A", "Conference Room B", None]),
        })
    
    df = pd.DataFrame(meetings)
    df = df.sort_values("start_time")
    return df


def generate_document_data(person_ids: list, num_days: int) -> pd.DataFrame:
    """Generate document collaboration data."""
    print(f"Generating document data for {num_days} days...")
    
    documents = []
    num_docs = len(person_ids) * 3  # ~3 documents per person
    
    doc_types = ["wiki", "doc", "presentation", "spreadsheet", "design_file"]
    platforms = ["sharepoint", "google_drive", "confluence", "notion"]
    
    for i in range(num_docs):
        # 1-3 authors
        num_authors = random.randint(1, 3)
        authors = random.sample(person_ids, num_authors)
        
        created_at = fake.date_time_between(
            start_date=f"-{num_days}d", end_date="now"
        )
        
        # Last modified 0-30 days after creation
        last_modified = created_at + timedelta(
            days=random.randint(0, min(30, (datetime.now() - created_at).days))
        )
        
        # Some documents have editors
        num_editors = random.randint(0, 2) if random.random() > 0.5 else 0
        editors = random.sample([p for p in person_ids if p not in authors], num_editors) if num_editors > 0 else []
        
        documents.append({
            "id": f"doc{i:05d}",
            "title": fake.sentence(nb_words=4),
            "author_ids": ",".join(authors),
            "editor_ids": ",".join(editors) if editors else "",
            "created_at": created_at.isoformat(),
            "last_modified": last_modified.isoformat(),
            "document_type": random.choice(doc_types),
            "platform": random.choice(platforms),
            "url": fake.url(),
            "size": random.randint(1000, 1000000),
        })
    
    df = pd.DataFrame(documents)
    df = df.sort_values("created_at")
    return df


def generate_code_data(person_ids: list, num_days: int) -> pd.DataFrame:
    """Generate code repository/commit data."""
    print(f"Generating code data for {num_days} days...")
    
    commits = []
    num_commits = len(person_ids) * 10  # ~10 commits per person
    
    repositories = ["backend-api", "frontend-app", "mobile-app", "infrastructure", "data-pipeline"]
    file_extensions = [".py", ".ts", ".tsx", ".js", ".java", ".go", ".rs"]
    
    for i in range(num_commits):
        author = random.choice(person_ids)
        repo = random.choice(repositories)
        
        # 1-5 files changed
        num_files = random.randint(1, 5)
        file_paths = [
            f"src/{fake.word()}{random.choice(file_extensions)}"
            for _ in range(num_files)
        ]
        
        timestamp = fake.date_time_between(
            start_date=f"-{num_days}d", end_date="now"
        )
        
        # Some commits have reviewers
        num_reviewers = random.randint(0, 2) if random.random() > 0.6 else 0
        reviewers = random.sample([p for p in person_ids if p != author], num_reviewers) if num_reviewers > 0 else []
        
        commits.append({
            "id": f"commit{i:05d}",
            "author_id": author,
            "repository": repo,
            "file_paths": ",".join(file_paths),
            "timestamp": timestamp.isoformat(),
            "reviewer_ids": ",".join(reviewers) if reviewers else "",
            "is_merge": random.random() > 0.8,
            "commit_hash": fake.sha1(),
            "message": fake.sentence(nb_words=5),
        })
    
    df = pd.DataFrame(commits)
    df = df.sort_values("timestamp")
    return df


def main():
    """Generate all sample data files."""
    print("=" * 60)
    print("Generating Sample Organizational Network Data")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate HRIS data first (required)
    hris_df = generate_hris_data(NUM_PEOPLE)
    person_ids = hris_df["person_id"].tolist()
    
    # Generate other data sources
    email_df = generate_email_data(person_ids, NUM_DAYS)
    slack_df = generate_slack_data(person_ids, NUM_DAYS)
    calendar_df = generate_calendar_data(person_ids, NUM_DAYS)
    documents_df = generate_document_data(person_ids, NUM_DAYS)
    code_df = generate_code_data(person_ids, NUM_DAYS)
    
    # Save to CSV files
    print("\nSaving data files...")
    hris_df.to_csv(OUTPUT_DIR / "hris.csv", index=False)
    email_df.to_csv(OUTPUT_DIR / "email.csv", index=False)
    slack_df.to_csv(OUTPUT_DIR / "slack.csv", index=False)
    calendar_df.to_csv(OUTPUT_DIR / "calendar.csv", index=False)
    documents_df.to_csv(OUTPUT_DIR / "documents.csv", index=False)
    code_df.to_csv(OUTPUT_DIR / "code.csv", index=False)
    
    print(f"\n✅ Sample data generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  - hris.csv: {len(hris_df)} people")
    print(f"  - email.csv: {len(email_df)} emails")
    print(f"  - slack.csv: {len(slack_df)} messages")
    print(f"  - calendar.csv: {len(calendar_df)} meetings")
    print(f"  - documents.csv: {len(documents_df)} documents")
    print(f"  - code.csv: {len(code_df)} commits")


if __name__ == "__main__":
    main()

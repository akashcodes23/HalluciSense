# ADR-003: Database Design and Data Isolation

## Status
Accepted

## Context
HalluciSense will process and store sensitive chat histories and verification reports. As a multi-tenant SaaS application, we must guarantee that User A can never accidentally (or maliciously) read User B's data. Additionally, we need to handle structured data (Messages, Users) alongside semi-structured data (Pillar Summaries, Analytics Payloads).

## Decision
- **Relational Database**: We will use PostgreSQL 16. It offers strong ACID guarantees and handles complex relational queries better than NoSQL alternatives for our use case (e.g., fetching a User's Chats, Messages, and Verification Reports).
- **Primary Keys**: We will use UUIDv4 for all primary keys instead of auto-incrementing integers to prevent ID-guessing attacks (Insecure Direct Object Reference) and to support distributed generation in the future.
- **Semi-Structured Data**: We will use PostgreSQL `JSONB` columns for data that lacks a rigid schema, such as the `pillar1_summary` or `raw_logits`. This prevents us from having to normalize highly variable data into overly complex table structures.
- **Data Isolation**: Application-level isolation will be enforced via the Repository pattern (e.g., always querying `WHERE user_id = :user_id`). Future implementations will consider PostgreSQL Row-Level Security (RLS) for defense-in-depth.

## Consequences
- **Positive**: High data integrity. `JSONB` provides massive flexibility without sacrificing the benefits of a relational DB for core entities. UUIDs enhance security.
- **Negative**: UUIDs are slightly larger and slower to index than 4-byte integers, but this performance hit is negligible at our projected scale compared to the security benefits.

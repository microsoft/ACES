# Excytin Demo - Incident Response Domain

**Cybersecurity incident response benchmark with database forensics and SQL analysis**

## Overview

The Excytin Demo domain simulates real-world cybersecurity incident response scenarios where agents must analyze security incidents by querying forensic databases. This domain tests an agent's ability to:

- Perform database forensics on MySQL databases containing security logs
- Analyze command and control (C2) behaviors and malware infections
- Investigate ransomware and credential theft incidents
- Identify threat actor activity groups and associated indicators of compromise (IOCs)
- Navigate complex database schemas with 20+ security data tables

The domain uses a **dual-environment architecture**:
- **Sandbox environment**: Python-based execution environment with MySQL client tools
- **Permanent services**: Persistent MySQL database (`incident_5_db`) containing forensic data

## Key Components

### Domain Structure

```
excytin_demo/
├── domain.yaml              # Domain configuration and metadata
├── client/
│   └── saber.yaml          # Client evaluation configuration
├── docker/
│   ├── Dockerfile.server   # Server container with domain logic
│   ├── Dockerfile.sandbox  # Sandbox with MySQL client tools
│   ├── build-images.sh     # Image build automation
│   └── db/
│       └── Dockerfile.incident_5  # MySQL database with forensic data
└── server/
    ├── config/
    │   ├── tasks/          # Task definitions (YAML)
    │   ├── prompts/        # Agent instructions and evaluation prompts
    │   └── environments/   # Docker Compose for sandbox and permanent services
    └── data/               # SQL dumps and incident artifacts
```

### Task Categories

**Incident 5 Series**: Command and Control (C2) investigation
- `incident_5_task_1`: Identify IP address of Manatee Tempest threat actor group
- `incident_5_task_2`: Analyze suspicious process command lines from user behavior changes
- `incident_5_task_3`: Investigate credential theft tool (Mimikatz) execution

### Evaluation Strategies

Tasks use two evaluation approaches:
- **LLM Judge**: GPT-4 evaluates agent responses against golden answers
- **Static Matching**: Exact string matching for deterministic answers

### Available Tools

Agents have access to containerized tools via MCP:
- **MySQL Client**: Database query and analysis (`mysql`, `pymysql`)
- **Network Tools**: Connectivity testing (`curl`, `ncat`, `dig`)
- **Python**: Scripting and data processing
- **Forensics**: Wireshark tools for packet analysis

## Quick Start

### Prerequisites

- SABER framework installed (see [main README](../../README.md))
- Docker and Docker Compose
- Azure OpenAI API credentials configured in `.env`

### Data Setup

The forensic database data is distributed as a zip archive (`data.zip`) and must be extracted before running the domain. Download or locate `data.zip` in the `domains/excytin_demo/` directory, then extract it in place with `unzip data.zip -d .` from within `domains/excytin_demo/` — this will create the `data/` folder containing the SQL dumps and incident artifacts required by the database Docker image.

After extraction, the `data/` directory should look like this:

```
data/
├── csv_files/
│   └── incident_5/
│       ├── AADManagedIdentitySignInLogs.csv
│       ├── AADManagedIdentitySignInLogs.meta
│       ├── DeviceEvents.csv
│       ├── DeviceEvents.meta
│       ├── ...                          # 38 CSV + 38 META files total
│       └── UrlClickEvents.meta
├── evaluations/
│   └── <session-uuid>/
│       └── incident_5_task_N/           # Reference evaluation results
└── sql_files/
    └── incident_5.sql                   # SQL dump loaded by the DB container
```

### Running the Domain

**This domain has been migrated to Inspect AI integration.** It's now available as a built-in domain in SABER installations.

**From any workspace with SABER installed:**

**1. Discover the domain:**
```bash
uv run inspect list tasks | grep excytin_demo
# Output: domains/excytin_demo/excytin_demo.py@excytin_demo
```

**2. Run all tasks:**
```bash
# Auto-builds images, auto-starts server, runs evaluation
uv run inspect eval domains/excytin_demo --model openai/gpt-4
```

**3. Run specific tasks:**
```bash
# Run all incident_5 tasks
uv run inspect eval domains/excytin_demo \
  -T task_filter=incident_5_* \
  --model openai/gpt-4

# Run a single task
uv run inspect eval domains/excytin_demo \
  -T task_filter=incident_5_task_1 \
  --model openai/gpt-4
```

**4. Advanced options:**
```bash
# Rebuild all Docker images
uv run inspect eval domains/excytin_demo \
  -T rebuild=true \
  --model openai/gpt-4

# Rebuild only the sandbox image
uv run inspect eval domains/excytin_demo \
  -T rebuild=sandbox \
  --model openai/gpt-4

# Use copilot agent
uv run inspect eval domains/excytin_demo \
  -T agent=copilot \
  --model openai/gpt-4

# Keep permanent services alive after eval
uv run inspect eval domains/excytin_demo \
  -T keep_permanent=true \
  --model openai/gpt-4

# Validate compose files before running
uv run inspect eval domains/excytin_demo \
  -T run_preflight=true \
  --model openai/gpt-4
```

### Task Parameters (`-T` Flags)

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_filter` | `str` | `None` | Glob or comma-separated task name filter (e.g., `incident_5_*`) |
| `dataset` | `str` | From `global.yaml` | Named task group selector |
| `agent` | `str` | `"react"` | Agent implementation: `react`, `copilot`, `claude_code` |
| `rebuild` | `str\|bool` | `None` | `true` → rebuild all images; `"sandbox"` → rebuild specific image |
| `run_preflight` | `bool` | `false` | Validate compose files before evaluation |
| `keep_permanent` | `bool` | `false` | Keep permanent Docker services (database) alive after eval |

#### Agent-Specific Parameters

| Parameter | Agent | Default | Description |
|-----------|-------|---------|-------------|
| `persona_file` | copilot, claude_code | `None` | Path to agent persona markdown file |
| `skills_dir` | copilot, claude_code | `None` | Path to skills directory (uploaded into sandbox) |
| `timeout` | copilot, claude_code | `300` | Agent execution timeout in seconds |
| `max_steps` | copilot, claude_code | `50` | Max tool calls before forced completion |
| `disallowed_tools` | claude_code | `""` | Comma-separated tools to disallow |

#### Docker / Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sandbox_compose` | `str` | `compose/sandbox.compose.yml` | Sandbox compose file path |
| `permanent_compose` | `str` | `None` | Permanent services compose file path |
| `permanent_project` | `str` | `saber-permanent` | Compose project name for permanent services |

### SABER CLI

Manage Docker environments outside of evaluations:

```bash
# Build images for excytin_demo
uv run saber build excytin_demo

# Force rebuild all images
uv run saber build excytin_demo --rebuild

# Rebuild only the sandbox image
uv run saber build excytin_demo --rebuild --image sandbox

# Start permanent services (database)
uv run saber start excytin_demo

# Tear down all excytin_demo Docker resources
uv run saber teardown excytin_demo --yes
```

### Inspecting Results

Evaluation logs are stored in:
```
logs/excytin/
├── <session-id>/         # Per-session evaluation logs
└── client-logs/          # Client-side execution logs
```

## Development and Contributing

### Domain Configuration

**Task Definition** (`server/config/tasks/incident_5/incident_5_1.yaml`):
```yaml
tasks:
  - task_id: "incident_5_task_1"
    description: "What is the IP address associated with Manatee Tempest?"
    sandbox_environment: "excytin_sandbox"
    initial_context:
      incident_context: "C2 behavior blocked on host..."
      database_connection:
        hostname: "saber-excytin-incident-5"
        database: "incident_5"
    evaluation_config:
      strategy: "llm_judge"
      criteria:
        golden_answer: "IP: 198.43.121.209"
```

**Client Configuration** (`client/saber.yaml`):
```yaml
server:
  mode: auto  # URLs auto-configured by inspect_ai task

tasks:
  task_ids: ["incident_5_task_1"]

agents:
  - id: "inspect_react"
    model: "openai/azure/gpt-4.1"
    tasks: ["incident_5_task_1"]

max_parallel_tasks: 2
```

### Adding New Tasks

1. **Create task YAML** in `server/config/tasks/incident_5/`:
   ```yaml
   tasks:
     - task_id: "incident_5_task_4"
       title: "New Investigation Task"
       description: "Your task description"
       sandbox_environment: "excytin_sandbox"
       evaluation_config:
         strategy: "llm_judge"  # or "static"
   ```

2. **Update client configuration** in `client/saber.yaml`:
   ```yaml
   tasks:
     task_ids: ["incident_5_task_1", "incident_5_task_4"]
   ```

3. **Test the new task**:
   ```bash
   uv run inspect eval domains/excytin_demo \
     -T task_filter=incident_5_task_4 \
     -T rebuild=true \
     --model openai/gpt-4
   ```

### Adding Custom Tools

Modify `docker/Dockerfile.sandbox` to add new tools:
```dockerfile
# Install additional forensics tools
RUN apt-get update && \
    apt-get install -y \
        postgresql-client \
        volatility3 \
        && rm -rf /var/lib/apt/lists/*
```

Then rebuild:
```bash
uv run inspect eval domains/excytin_demo \
  -T rebuild=sandbox \
  --model openai/gpt-4
```

### Database Schema Exploration

The `incident_5` database contains:
- Security event logs (DeviceEvents, NetworkEvents, ProcessEvents)
- User activity data (UserBehavior, AccountActivity)
- Threat intelligence (ThreatIndicators, MalwareDetections)

Agents must explore the schema to understand table relationships and complete investigations.

### Evaluation Customization

**LLM Judge Configuration** (`server/config/prompts/judge/`):
- `cybersecurity_incident_system.j2`: Judge system prompt
- `cybersecurity_incident_user.j2`: Evaluation criteria template

**Static Evaluation**:
```yaml
evaluation_config:
  strategy: static
  criteria:
    expected_answers:
      - "198.43.121.209"
  scoring:
    max_score: 1.0
```

### Testing Best Practices

1. **Test with rebuild for clean state**:
   ```bash
   uv run inspect eval domains/excytin_demo \
     -T rebuild=true \
     --model openai/gpt-4
   ```

2. **Tear down Docker resources after testing**:
   ```bash
   uv run saber teardown excytin_demo --yes
   ```

3. **Monitor server logs**:
   ```bash
   docker logs saber-excytin-demo-server -f
   ```

4. **Inspect database directly**:
   ```bash
   docker exec -it saber-excytin-incident-5 mysql -u incident_5_user -p incident_5
   ```

### Contributing Guidelines

When contributing to the Excytin domain:

- ✅ **Task YAML files**: Follow existing format in `server/config/tasks/`
- ✅ **Evaluation criteria**: Provide clear golden answers or rubrics
- ✅ **Database consistency**: Ensure forensic data is realistic and well-structured
- ✅ **Prompt templates**: Use Jinja2 templates in `server/config/prompts/`
- ✅ **Documentation**: Update this README when adding new task categories
- ✅ **Testing**: Run full evaluation suite before submitting changes

### Troubleshooting

**Database connection issues:**
```bash
# Check if database container is running
docker ps | grep incident-5

# Verify database network
docker network inspect excytin_demo-shared-network
```

**Agent can't find tables:**
- Ensure database is properly seeded (check `server/data/` for SQL dumps)
- Verify `initial_context.database_connection` in task YAML

**Evaluation failures:**
- Check LLM judge model availability in `.env`
- Review judge prompts in `server/config/prompts/judge/`
- Verify golden answers match expected format

---

**Domain Maintainer**: saber-team@security.com | **SABER Docs**: [../../docs/](../../docs/) | **Main README**: [../../README.md](../../README.md)

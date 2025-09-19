# Excytin Cybersecurity Incident Response Domain

## Mission Overview

The `excytin_demo` domain is a comprehensive **cybersecurity incident response simulation** that evaluates autonomous security agents against realistic digital forensics and threat hunting scenarios. This domain models a compromised enterprise network environment where agents must investigate security incidents, analyze forensic data, and identify threat actor activities.

### Core Mission Objectives

1. **Digital Forensics Investigation**: Agents perform systematic database analysis of security logs to reconstruct attack timelines
2. **Threat Actor Attribution**: Identify specific Advanced Persistent Threat (APT) groups and their tactics, techniques, and procedures (TTPs)
3. **Incident Response Workflow**: Follow structured investigation procedures to answer critical security questions
4. **Evidence Collection**: Extract precise technical indicators from forensic databases
5. **Attack Chain Reconstruction**: Trace lateral movement, credential theft, and payload deployment activities

## Threat Landscape & Attack Scenarios

This domain simulates a sophisticated multi-stage cyberattack involving:

### **Primary Threat Actor: Manatee Tempest APT Group**
- **C2 Infrastructure**: 198.43.121.209 (vectorsandarrows.com)
- **Attack Vector**: Initial compromise via command-and-control communications
- **Target Network**: vnevado.alpineskihouse.co domain

### **Attack Kill Chain Components:**
1. **Initial Access**: C2 beaconing from compromised hosts (`vnevado-win10r`, `vnevado-win11u`)
2. **Credential Harvesting**: Mimikatz deployment for LSASS memory dumping
3. **Lateral Movement**: PSExec-based movement to jump servers (`Vnevado-jump`)
4. **Privilege Escalation**: Pass-the-Hash attacks using stolen NTLM credentials
5. **Ransomware Deployment**: Remote payload execution on target systems
6. **Persistence**: Azure AD token theft attempts (Primary Refresh Token access)

### **Investigation Scenarios (1 Example)**
- **incident_5_task_1**: Manatee Tempest IP attribution investigation

## Technical Architecture

### **Incident Response Infrastructure**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Cybersecurity Incident Response Domain (excytin_demo)                  │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ SABER Client (Security Analyst Agent)                              ││
│  │  ├── React Agent with MCP Tools                                    ││
│  │  ├── Database Forensics Capabilities                               ││
│  │  ├── SQL Query Construction & Analysis                             ││
│  │  └── Structured Investigation Workflow                             ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                │                                        │
│                    ┌───────────────────────────┐                       │
│                    │ SABER Server               │                       │
│                    │  ├─ Episode Management    │                       │
│                    │  ├─ Task Orchestration    │                       │
│                    │  ├─ MCP Tool Server       │                       │
│                    │  └─ Evaluation Framework  │                       │
│                    └───────────────────────────┘                       │
│                                │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Forensics Investigation Environment                                 ││
│  │  ┌─────────────────────┐    ┌─────────────────────────────────────┐ ││
│  │  │ Sandbox Container   │    │ Incident Database (MySQL)          │ ││
│  │  │ - Linux Environment │    │ - 20+ Security Tables              │ ││
│  │  │ - MySQL Client      │    │ - Enterprise Logs                  │ ││
│  │  │ - Investigation     │◄──►│ - Attack Timeline Data             │ ││
│  │  │   Tools             │    │ - Forensic Artifacts               │ ││
│  │  │ - Persistent Access │    │ - Network Traffic Logs             │ ││
│  │  └─────────────────────┘    └─────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### **Investigation Workflow**
1. **Episode Initialization**: Agent receives incident context and investigation question
2. **Database Discovery**: Systematic exploration of forensics database schema (env_monitor_db)
3. **Evidence Collection**: SQL-based analysis of security logs and artifacts
4. **Pattern Recognition**: Correlation of events to identify attack patterns
5. **Technical Attribution**: Extraction of precise IOCs, timestamps, and technical details
6. **Evaluation**: LLM-based judging of investigation methodology and findings accuracy

### **Forensics Database Schema**
- **Core Tables**: 20+ interconnected security tables containing enterprise logs
- **Log Sources**: Windows Event Logs, Network Traffic, Process Execution, Authentication Events
- **Data Scope**: Multi-host environment with jump servers, workstations, and domain controllers
- **Time Range**: Attack timeline spanning weeks with precise timestamp correlation
- **Artifact Types**: Process command lines, network connections, file operations, registry changes

## Quick Start Guide

### Prerequisites

**Required Components:**
- Docker and Docker Compose installed
- Python with `uv` package manager  
- Access to Azure OpenAI or equivalent LLM endpoint
- **Excytin Forensics Data** (contact Kyle DeProw or Anand Mudgerikar)

**Critical**: You must obtain the Excytin incident data files and place them at:
```
server/data/
├── csv_files/incident_5/*.csv
└── sql_files/incident_5.sql
```

### 1. Build Forensics Infrastructure

```bash
cd /path/to/SABER/domains/excytin_demo/docker
./build-images.sh --full-build
```

**Built Images:**
- `saber/excytin-server:latest` - SABER investigation orchestration server
- `saber/excytin-client:latest` - Security analyst agent container  
- `saber/excytin-sandbox:latest` - Forensics investigation environment
- `saber/excytin-incident-5:latest` - Pre-loaded MySQL database with incident data

### 2. Configure Investigation Environment

```bash
# Configure LLM credentials
cp client/.env.template client/.env
# Edit client/.env with your Azure OpenAI endpoint details

# Launch incident response infrastructure
docker compose up -d
```

**Active Services:**
- `saber-excytin-server` (ports 8000/8001) - Investigation server
- `saber-excytin-incident-5` - Persistent forensics database
- `saber-excytin-client` - Agent execution environment

### 3. Verify Forensics Database

```bash
# Check database connectivity and tables
docker logs saber-excytin-incident-5 | grep "ready for connections"

# Verify forensics data loading
docker exec saber-excytin-incident-5 mysql -u admin -padmin env_monitor_db -e "SHOW TABLES;"
```

### 4. Execute Investigation

#### **Single Investigation (Recommended)**
```bash
cd client/
./run_demo.sh
```

#### **Multiple Incident Analysis**
```bash
# Edit client/saber.yaml to include multiple task_ids:
# tasks:
#   task_ids: ["incident_5_task_1", "incident_5_task_2", "incident_5_task_3"]

./run_demo.sh --verbose
```

#### **Investigation Results Analysis**
```bash
# Launch investigation results viewer
uv run python -m saber.client inspect view --log-dir ./client/logs --no-browser --host 0.0.0.0 --port 7577

# Browser URL: http://0.0.0.0:7577
```

## Expected Investigation Behavior

### **Successful Investigation Workflow:**
1. **Session Establishment**: Agent connects to SABER investigation server
2. **Task Assignment**: Receives specific incident investigation assignment with context
3. **Database Discovery**: Explores forensics database schema (`env_monitor_db`)
4. **Evidence Gathering**: Executes systematic SQL queries across security tables
5. **Pattern Analysis**: Correlates events to reconstruct attack timeline  
6. **IOC Extraction**: Identifies precise technical indicators (IPs, hostnames, SIDs, command lines)
7. **Final Report**: Submits findings for expert evaluation

### **Agent Investigation Capabilities:**
- ✅ **Forensics Database Access**: Direct MySQL connectivity to incident data
- ✅ **SQL Query Construction**: Dynamic query building based on investigation needs  
- ✅ **Schema Exploration**: SHOW TABLES, DESCRIBE commands for database discovery
- ✅ **Log Correlation**: Cross-referencing multiple tables for timeline reconstruction
- ✅ **Technical Precision**: Exact matching of technical indicators and artifacts
- ✅ **Investigation Methodology**: Structured approach following cybersecurity best practices

### **Evaluation Standards:**
- **Technical Accuracy**: Exact match for IPs (198.43.121.209), hostnames (vnevado-win10r), SIDs
- **Investigation Quality**: SQL techniques, database exploration approach, logical progression
- **Evidence Correlation**: Ability to connect related security events across tables
- **IOC Identification**: Precise extraction of indicators of compromise
- **Timeline Reconstruction**: Understanding of attack sequence and lateral movement

## Investigation Diagnostics & Troubleshooting

### **Forensics Database Validation**
```bash
# Verify incident database is loaded with forensics data
docker exec saber-excytin-incident-5 mysql -u admin -padmin env_monitor_db -e "
  SELECT COUNT(*) as total_tables FROM information_schema.tables 
  WHERE table_schema='env_monitor_db';"

# Check forensics data volume (should show 20+ tables with substantial data)
docker exec saber-excytin-incident-5 mysql -u admin -padmin env_monitor_db -e "
  SELECT table_name, table_rows 
  FROM information_schema.tables 
  WHERE table_schema='env_monitor_db' AND table_rows > 0
  ORDER BY table_rows DESC;"
```

### **Investigation Progress Monitoring**
```bash
# Monitor agent investigation in real-time
docker logs saber-excytin-server | grep -E "(episode|investigation|query|evidence)"

# Check episode container creation (investigation environments)
docker ps -a | grep "excytin-sandbox"

# Review investigation methodology in agent logs
tail -f ./client/logs/saber_client_*/saber_client.log | grep -E "(Thought|Action|mysql)"
```

### **Common Investigation Issues**

#### **Database Connectivity Problems**
```bash
# Test database connection from sandbox environment
docker exec $(docker ps -q --filter "name=excytin-sandbox") \
  mysql -h saber-excytin-incident-5 --skip-ssl -u admin -padmin env_monitor_db -e "SHOW TABLES;"
```

#### **Investigation Quality Issues**
- **Symptom**: Agent returns generic answers or fails to find evidence
- **Diagnosis**: Check if agent is properly exploring database schema
- **Solution**: Verify agent uses systematic SQL exploration (SHOW TABLES, DESCRIBE)

#### **Evaluation Failures**  
- **Symptom**: LLM judge gives low scores despite correct answers
- **Diagnosis**: Review investigation methodology vs. just final answers
- **Solution**: Ensure agent demonstrates proper forensics analysis techniques

## Configuration Reference

### **Investigation Tasks (`server/config/tasks/`)**

#### **Global Configuration (`global.yaml`)**
```yaml
domain: excytin_demo
permanent_environment: incident_5  # Persistent forensics database

global_defaults:
  execution_config:
    allowed_executors: [bash, python]  # Investigation tool access
    timeout: 30                        # Query timeout seconds
  episode_config:
    max_steps: 15                      # Investigation step limit
  benchmark_config:
    episode_attempts: 1                # Single investigation attempt
```

#### **Incident Tasks (`incident_5/*.yaml`)**
Each investigation task includes:
- **Task Context**: Specific incident scenario and background
- **Investigation Question**: Precise technical question requiring evidence
- **Evaluation Strategy**: Static exact-match or LLM-based methodology assessment  
- **Forensics Database**: Connection details to `env_monitor_db`
- **Subtask Checkpoints**: Investigation milestones for progress tracking

**Example Task Structure:**
```yaml
tasks:
  - task_id: "incident_5_task_1"
    description: "What is the IP address associated with the Manatee Tempest activity group?"
    initial_context:
      incident_context: "C2 behavior blocked on vnevado-win10r..."
      question: "Identify Manatee Tempest IP address"
      database_connection:
        hostname: "incident-5-db"
        database: "env_monitor_db"
        username: "admin"
        password: "admin"
```

### **Forensics Environment (`server/config/environments/`)**

#### **Persistent Database (`permanent/incident_5.compose.yml`)**
- **Service**: `saber-excytin-incident-5` MySQL container
- **Data**: Pre-loaded with 20+ forensics tables
- **Network**: Shared `excytin-database` network for investigation access
- **Persistence**: Runs continuously during server lifetime

#### **Investigation Sandbox (`sandbox/excytin_sandbox.compose.yml`)**  
- **Service**: `excytin-sandbox` per-episode container
- **Tools**: MySQL client, bash, python for forensics analysis
- **Isolation**: Episode-specific containers with unique IDs
- **Resources**: 512MB memory, 0.5 CPU limits

### **Agent Configuration (`client/saber.yaml`)**

#### **Investigation Agent Setup**
```yaml
model: "openai/azure/gpt-4.1"           # LLM for investigation reasoning
agent:
  id: "inspect_react"                    # React agent for step-by-step analysis
  debug_mode: false

tasks:
  task_ids: ["incident_5_task_1"]        # Specific investigation assignments
  task_type: "individual"               # Individual incident investigations

server:
  rest_url: "http://saber-excytin-server:8000"  # Investigation orchestration
  mcp_url: "http://saber-excytin-server:8001"   # Tool access endpoint
```

### **Database Schema Reference**

#### **Forensics Database Structure (`env_monitor_db`)**
The incident database contains 20+ interconnected tables with:
- **Process Execution Logs**: Command lines, process IDs, timestamps
- **Network Traffic**: Source/destination IPs, protocols, URLs
- **Authentication Events**: User SIDs, login activities, privilege changes
- **File System Activity**: File access, modifications, creation timestamps
- **Registry Operations**: Windows registry key changes and access
- **Memory Analysis**: Process memory dumps, LSASS interactions
- **Lateral Movement**: PSExec activities, remote execution traces

#### **Investigation Connection Pattern**
```sql
-- Standard connection from investigation sandbox
mysql -h incident-5-db --skip-ssl -u admin -padmin env_monitor_db

-- Database exploration workflow
SHOW TABLES;                              -- Discover available forensics tables
DESCRIBE table_name;                      -- Analyze table schema
SELECT * FROM table_name LIMIT 10;       -- Sample data examination
```

## Advanced Investigation Scenarios

### **Multi-Vector Attack Analysis**
For comprehensive incident response training, configure multiple tasks:

```yaml
# client/saber.yaml - Full incident investigation suite
tasks:
  task_ids: [
    "incident_5_task_1",   # Manatee Tempest IP attribution
    "incident_5_task_2",   # Suspicious process identification  
    "incident_5_task_3",   # Credential theft tool analysis
    "incident_5_task_21",  # PSExec lateral movement
    "incident_5_task_22",  # LSASS memory modification
    "incident_5_task_23",  # C2 URL extraction
    "incident_5_task_41"   # Compromised user SID identification
  ]
```

### **Investigation Methodology Evaluation**
The domain supports two evaluation approaches:

#### **Static Evaluation** (Exact Technical Matching)
```yaml
evaluation_config:
  strategy: "static"
  criteria:
    expected_answers:
      - "198.43.121.209"           # Exact IP match required
      - "S-1-5-21-1840191660-8534830288-125585561-1522"  # Exact SID
```

#### **LLM Judge Evaluation** (Methodology Assessment)
```yaml
evaluation_config:
  strategy: "llm_judge"
  criteria:
    golden_answer: "Agent should identify Manatee Tempest IP as 198.43.121.209"
    model: "openai/azure/gpt-4.1"
    judge_system_template: "cybersecurity_incident_system.md"
    judge_user_template: "cybersecurity_incident_user.md"
```

### **Forensics Data Requirements**

#### **Data Structure** (Must be obtained separately)
```
server/data/
├── sql_files/
│   └── incident_5.sql          # Database schema and initial data
└── csv_files/incident_5/
    ├── process_logs.csv        # Process execution forensics
    ├── network_traffic.csv     # Network communication logs  
    ├── auth_events.csv         # Authentication and privilege events
    ├── file_operations.csv     # File system activity logs
    └── [additional_tables].csv # 15+ additional forensics tables
```

#### **Data Volume Expectations**
- **Total Tables**: 20+ interconnected security tables
- **Log Entries**: Thousands of forensics records per table
- **Time Span**: Multi-week attack timeline with precise timestamps
- **Data Sources**: Enterprise Windows environment with multiple hosts

### **Performance Tuning**

#### **Investigation Timeout Configuration**
```yaml
# Adjust for complex forensics queries
global_defaults:
  execution_config:
    timeout: 60  # Increase for large database queries
  episode_config:
    max_steps: 25  # Allow deeper investigation
```

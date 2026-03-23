# SonarQube

## 1) Purpose
Use SonarQube for static analysis, code-smell/security checks, duplication detection, and quality-gate enforcement.

## 2) Local Setup
1. Start SonarQube server and database:
   - `./scripts/sonarqube-up.sh`
   - (or) `docker compose -f docker-compose.sonarqube.yml up -d`
2. Create a user token in SonarQube UI.
3. Export token locally:
   - `export SONAR_TOKEN="<your-token>"`

This repo is already configured via `sonar-project.properties`.

## 3) Local Scan (Project Root)
```bash
./venv/bin/pytest --cov --cov-report=xml:coverage.xml -q
sonar-scanner "-Dsonar.host.url=http://localhost:9000" "-Dsonar.token=$SONAR_TOKEN"
```

Notes:
- Always generate `coverage.xml` before scanning.
- Run from repository root so `sonar-project.properties` is picked up.
- In `zsh`, quote `-D` values that contain glob patterns.

## 4) CI Integration
CI workflow `.github/workflows/ci.yml` includes job `sonar-quality-gate` that:
1. installs dependencies,
2. runs full tests with coverage xml,
3. runs Sonar scanner in Docker (`sonarsource/sonar-scanner-cli`).

Required repository settings:
- secret: `AWS_GITHUB_ACTIONS_ROLE_ARN`
- variable: `SONAR_HOST_URL`
- variable: `SONAR_AWS_SECRET_ID` (AWS Secrets Manager secret id containing Sonar token JSON)
- variable: `SONAR_TOKEN_JSON_KEY` (optional; defaults to `SONAR_TOKEN`)

The workflow uses OIDC to assume AWS role and fetches the Sonar token from AWS Secrets Manager at runtime.
No Sonar token is stored in repository code.

## 5) Stop Local SonarQube
```bash
./scripts/sonarqube-down.sh
```

## 6) Troubleshooting
- `command not found: sonar-scanner`
  - install scanner locally (`brew install sonar-scanner` on macOS) or use CI job.
- `Communicating with SonarQube Cloud` + `403`
  - scanner is targeting SonarCloud; pass your self-hosted URL with `-Dsonar.host.url=...`.
- coverage gate fails near threshold
  - inspect New Code coverage in Sonar and add tests for low-coverage touched files first.

# GEMINI.md

## Project Overview

This project, named **InferFlux**, is a high-performance inference server written in C++. It is designed to be a drop-in replacement for services like LM Studio and Ollama, with a strong focus on enterprise-grade features and performance.

The server is built with a modular architecture, supporting multiple backends for different hardware targets, including CPU, CUDA, ROCm, and Apple Metal (MPS). It can load models in GGUF and safetensors formats, either from local storage or from Hugging Face.

Key features of InferFlux include:

*   **Continuous Batching:** A paged KV cache and a sophisticated scheduler enable efficient handling of multiple concurrent requests.
*   **Speculative Decoding:** The server can use a smaller, faster "draft" model to generate candidate tokens, which are then verified by the primary model, potentially speeding up inference.
*   **OpenAI-Compatible API:** The server exposes an HTTP/1.1 REST API (with SSE streaming) compatible with the OpenAI API format, making it easy to integrate with existing tools and applications.
*   **Enterprise-Grade Security and Observability:**
    *   **Authentication:** Supports API keys and OIDC for secure access.
    *   **Authorization:** Implements Role-Based Access Control (RBAC) with different scopes for different API keys.
    *   **Rate Limiting:** Protects the server from being overloaded.
    *   **Audit Logging:** Records all requests and responses for security and compliance.
    *   **Metrics and Tracing:** Exposes a `/metrics` endpoint for Prometheus and supports OpenTelemetry for distributed tracing.
    *   **Policy Enforcement:** Includes a guardrail system for blocking certain prompts or responses, and can integrate with external OPA/Cedar policy engines.
*   **Command-Line Interface:** A powerful CLI tool, `inferctl`, is provided for interacting with the server, including an interactive chat mode.

## Building and Running

The project uses CMake for building. The following scripts are provided for convenience:

*   **Build:**
    ```bash
    ./scripts/build.sh
    ```
    This script will compile the server (`inferfluxd`), the CLI (`inferctl`), and the unit tests. The build artifacts are placed in the `build/` directory.

*   **Run:**
    ```bash
    ./scripts/run_dev.sh --config config/server.yaml
    ```
    This script starts the `inferfluxd` server with the configuration specified in `config/server.yaml`.

## Development Conventions

*   **Code Style:** The codebase follows modern C++17 conventions.
*   **Testing:** The project includes both unit and integration tests.
    *   **Unit Tests:**
        ```bash
        ctest --test-dir build --output-on-failure
        ```
    *   **Integration Tests:**
        ```bash
        ctest -R IntegrationSSE --output-on-failure
        ```
        Note: The integration tests require the `INFERFLUX_MODEL_PATH` environment variable to be set to the path of a GGUF model file.

*   **Dependencies:** The project uses several external libraries, including OpenSSL and `llama.cpp`. Dependencies are managed through a combination of `find_package` and submodules.

## Key Files

*   `README.md`: The main entry point for understanding the project.
*   `CMakeLists.txt`: The main CMake build script, which defines the project structure and dependencies.
*   `server/main.cpp`: The entry point for the `inferfluxd` server application.
*   `cli/main.cpp`: The implementation of the `inferctl` command-line interface.
*   `config/server.yaml`: The main configuration file for the server.
*   `docs/`: Contains additional documentation, including the project's architecture, design, and roadmap.
*   `scripts/`: Contains helper scripts for building and running the project.
*   `tests/`: Contains the unit and integration tests.

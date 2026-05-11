import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    # Create a simple Go service
    svc_dir = tmp_path / "services" / "user-service"
    svc_dir.mkdir(parents=True)

    (svc_dir / "main.go").write_text('''package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default()
    r.GET("/health", healthHandler)
    r.POST("/api/users", createUser)
    r.GET("/api/users/:id", getUser)

    api := r.Group("/api/v1")
    api.GET("/items", listItems)
    api.POST("/items", createItem)

    r.Run(":8080")
}

func healthHandler(c *gin.Context) {
    c.JSON(200, gin.H{"status": "ok"})
}

func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    c.JSON(201, gin.H{"id": "123"})
}
''')

    # Create a Python service
    py_dir = tmp_path / "services" / "auth-service"
    py_dir.mkdir(parents=True)

    (py_dir / "app.py").write_text('''from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/auth/login")
def login(email: str, password: str):
    return {"token": "abc"}

@app.get("/auth/verify")
def verify():
    # Call user service
    resp = requests.get("http://localhost:8080/api/users/123")
    return resp.json()
''')

    # Create a TS service
    ts_dir = tmp_path / "services" / "web-ui"
    ts_dir.mkdir(parents=True)
    (ts_dir / "tsconfig.json").write_text('{}')
    src_dir = ts_dir / "src"
    src_dir.mkdir()

    (src_dir / "server.ts").write_text('''import express from 'express';
const app = express();

app.get('/health', (req, res) => res.json({ status: 'ok' }));
app.post('/api/chat', async (req, res) => {
    const resp = await fetch('http://localhost:8080/api/users/123');
    res.json(await resp.json());
});

app.listen(3000);
''')

    # Create ports.yaml
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "ports.yaml").write_text('''ports:
  user-service:
    port: 8080
    service: user-service
  auth-service:
    port: 8081
    service: auth-service
  web-ui:
    port: 3000
    service: web-ui
''')

    return tmp_path

@pytest.fixture
def config(tmp_project):
    from architect_sim.config import Config
    return Config(str(tmp_project),
                  services_dir=str(tmp_project / "services"),
                  config_file=str(tmp_project / "config" / "ports.yaml"))

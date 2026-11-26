"""Exercise the bundled browser assets and simulation endpoints over loopback."""

import json
from contextlib import closing
from http.client import HTTPConnection
from http.server import HTTPServer
from threading import Thread

import pytest
import stim

from fdem.web import HOST, WorkbenchHandler


@pytest.fixture
def workbench_connection():
    """Start the real handler on an available local port."""
    with HTTPServer((HOST, 0), WorkbenchHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with closing(HTTPConnection(HOST, server.server_port, timeout=10)) as connection:
                yield connection
        finally:
            server.shutdown()
            thread.join()


def test_workbench_serves_its_assets(workbench_connection: HTTPConnection) -> None:
    """Serve the bundled HTML, stylesheet, and application script."""
    for path, content_type in (
        ("/", "text/html"),
        ("/style.css", "text/css"),
        ("/app.js", "text/javascript"),
    ):
        workbench_connection.request("GET", path)
        response = workbench_connection.getresponse()
        assert response.status == 200
        assert response.headers.get_content_type() == content_type
        assert response.read()


@pytest.mark.parametrize("kind", ["uniform", "one_liz", "two_liz"])
def test_workbench_returns_correlation_and_circuit(
    workbench_connection: HTTPConnection, kind: str
) -> None:
    """Verify a default-size run, its signed matrix, and its circuit export."""
    workbench_connection.request("GET", "/api/defaults")
    response = workbench_connection.getresponse()
    assert response.status == 200
    defaults = json.loads(response.read())
    settings = {"distance": 3, "rounds": 3, "shots": 2048, "seed": 42}
    if kind == "uniform":
        path = "/api/uniform-dem"
        settings["p"] = 0.01
    else:
        path = "/api/simulate"
        zones = defaults["zones"]
        if kind == "two_liz":
            zones = zones + zones[3:]
        settings.update(
            zones=zones,
            timing=defaults["timing"],
            noise=defaults["noise"],
            lookup_mode="strict",
            cooling={
                "resident_ids": [], "recool_us": None,
                "threshold": None, "after_measure_us": None,
            },
        )

    workbench_connection.request(
        "POST", path, json.dumps(settings), {"Content-Type": "application/json"}
    )
    response = workbench_connection.getresponse()
    result = json.loads(response.read())
    assert response.status == 200, result
    assert result["shots"] == settings["shots"]
    count = result["detectors"]
    assert count == 24
    matrix = result["matrix"]
    assert len(matrix) == count
    assert all(len(row) == count for row in matrix)
    assert all(0 <= matrix[index][index] <= 1 for index in range(count))
    assert all(matrix[i][j] == matrix[j][i] for i in range(count) for j in range(i))
    assert len(result["checks"]) == 8
    assert stim.Circuit(result["stim"]).num_detectors == count
    if kind == "uniform":
        assert stim.DetectorErrorModel(result["dem"]).num_detectors == count
    else:
        assert len(result["schedules"]) == settings["rounds"]


def test_workbench_rejects_cross_origin_requests(workbench_connection: HTTPConnection) -> None:
    """Keep the local simulation routes accessible to the workbench's own origin."""
    workbench_connection.request("GET", "/api/defaults", headers={"Origin": "https://example.org"})
    response = workbench_connection.getresponse()
    assert response.status == 403
    response.read()

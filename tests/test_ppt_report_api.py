from pathlib import Path
import pytest
from flask import Flask
from tools.Ppt_report_Automation import routes

@pytest.fixture
def client(tmp_path):
    app = Flask(__name__)
    app.config.update(TESTING=True, OUTPUT_FOLDER=str(tmp_path))
    app.register_blueprint(routes.ppt_report_bp, url_prefix='/api/ppt-report')
    return app.test_client()

@pytest.mark.parametrize('body', [{}, [], {'project_id': True}, {'project_id': 1.2}, {'project_id': 1, 'session_ids': 'abc'}, {'project_id': 1, 'region': None}])
def test_bad_payload(client, body):
    assert client.post('/api/ppt-report/generate', json=body).status_code == 400


def test_generate_download(client, monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        Path(kwargs['output_path']).write_bytes(b'example-pptx')
    monkeypatch.setattr(routes, 'generate_ppt_for_project', generate)
    assert client.get('/api/ppt-report/health').status_code == 200
    assert client.get('/api/ppt-report/download/210').status_code == 404
    response = client.post('/api/ppt-report/generate', json={'project_id': 210, 'session_ids': '4479,4478'})
    assert response.status_code == 200
    assert calls[0]['session_ids'] == [4479, 4478]
    download = client.get(response.json['download_url'])
    assert download.data == b'example-pptx'
    assert 'Mobility_DT_Project_210.pptx' in download.headers['Content-Disposition']
    download.close()


def test_errors_preserve_report(client, monkeypatch):
    assert client.post('/api/ppt-report/generate', data='{}').status_code == 415
    assert client.post('/api/ppt-report/generate', data='{', content_type='application/json').status_code == 400
    with client.application.app_context():
        directory = routes._output_dir()
        directory.mkdir(parents=True)
        report = directory / 'Mobility_DT_Project_210.pptx'
        report.write_bytes(b'previous')
    def fail(**kwargs):
        raise RuntimeError('generation failed')
    monkeypatch.setattr(routes, 'generate_ppt_for_project', fail)
    assert client.post('/api/ppt-report/generate', json={'project_id': 210}).status_code == 500
    assert report.read_bytes() == b'previous'
    assert list(directory.iterdir()) == [report]

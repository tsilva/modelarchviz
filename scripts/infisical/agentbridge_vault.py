"""Opt-in Agent Vault app launcher; never retrieves application secrets."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = 'modelarchviz'
PROJECT = '8b4ace0c-47c7-49ed-be67-2e8df5f1b697'
ADAPTER = 'app/agentbridge.ts'
DEFAULT_MODEL = 'openai/gpt-6.1-terra'
PROFILE = Path.home() / '.config/infisical/agentbridge-clients' / (REPO + '-read.json')
MARKER = 'agentbridge_public_placeholder'
NODE_RELATIVE = '.config/infisical/runtimes/node-v24.22.1-darwin-arm64/bin/node'
FINGERPRINT = (
    'SHA256:37:B2:E0:81:45:EC:C0:58:46:2E:D1:2F:E1:2B:E6:AE:'
    '2B:78:B1:B3:F0:21:CD:8F:EC:54:CC:30:2D:36:9E:37'
)


def source_files():
    return {'scripts/infisical/' + name for name in (
        'agentbridge_vault.py', 'agentbridge_client.mjs', 'agentbridge_transport.mjs',
        'agentbridge_preload.mjs', 'agentbridge_app.mjs', 'agentbridge.sb', 'agentbridge_app.sb',
    )} | {'.infisical.json', ADAPTER, 'app/api/chat/route.ts', 'app/chat-backend.ts', 'next.config.mjs',
         'scripts/generate-model-artifacts.mjs', 'scripts/model-highlight-anchors.mjs',
         'instrumentation.ts', 'instrumentation-client.ts', 'sentry.server.config.ts',
         'package.json', 'pnpm-lock.yaml'}


def verify_app_dependencies():
    manifest = json.loads((ROOT / 'package.json').read_text())
    for name in ('next', 'react', 'react-dom'):
        expected = manifest['dependencies'][name]
        installed = ROOT / 'node_modules' / name / 'package.json'
        if (not isinstance(expected, str) or not re.fullmatch(r'\d+\.\d+\.\d+', expected)
                or not installed.is_file()):
            raise ValueError('Install the reviewed locked app dependencies before serving.')
        metadata = json.loads(installed.read_text())
        if metadata.get('name') != name or metadata.get('version') != expected:
            raise ValueError('Installed app dependencies differ from the reviewed manifest; refresh the locked environment before serving.')


def load_profile(mode='check'):
    profile = PROFILE if mode == 'check' else PROFILE.with_name(REPO + '-app.json')
    parent = profile.parent.lstat()
    if (profile.parent.resolve() != profile.parent or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid() or parent.st_mode & 0o077):
        raise ValueError('The AgentBridge profile directory must be canonical and owner-only.')
    if profile.is_symlink():
        raise ValueError('The AgentBridge profile must be a regular file.')
    info = profile.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('The AgentBridge profile must be owner-only.')
    p = json.loads(profile.read_text())
    fields = {'version', 'repository', 'repo_path', 'project_id', 'source_environment',
              'source_path', 'bundle', 'base_url', 'proxy', 'ca_file', 'ca_fingerprint',
              'infisical', 'node', 'adapter', 'adapter_sha256', 'default_model',
              'code_sha256', 'runtime_sha256'}
    if set(p) != fields or type(p['version']) is not int or p['version'] != 1:
        raise ValueError('Unsupported AgentBridge profile.')
    if (ROOT / '.infisical.yaml').exists():
        raise ValueError('A new native binding requires explicit review.')
    native = json.loads((ROOT / '.infisical.json').read_text())
    if native != {'workspaceId': PROJECT, 'defaultEnvironment': 'dev',
                  'gitBranchToEnvironmentMapping': None, 'domain': 'https://app.infisical.com'}:
        raise ValueError('The native source identity differs from the reviewed binding.')
    expected = {'repository': REPO, 'repo_path': str(ROOT), 'project_id': PROJECT,
        'source_environment': 'dev', 'source_path': '/',
        'bundle': 'local-' + REPO + '-agentbridge' + ('-read' if mode == 'check' else ''),
        'base_url': 'http://127.0.0.1:8082/api/v1', 'proxy': '127.0.0.1:8283',
        'adapter': ADAPTER, 'default_model': DEFAULT_MODEL,
        'infisical': str(Path.home() / '.config/infisical/cli/0.45.6/infisical'),
        'ca_file': str(Path.home() / '.config/infisical/agent-vault-ca.pem'),
        'ca_fingerprint': FINGERPRINT}
    if any(p[key] != value for key, value in expected.items()):
        raise ValueError('The service mapping differs from the reviewed binding.')
    adapter = ROOT / ADAPTER
    if adapter.is_symlink() or hashlib.sha256(adapter.read_bytes()).hexdigest() != p['adapter_sha256']:
        raise ValueError('The application routing helper changed; review the profile again.')
    if not re.fullmatch(r'SHA256:(?:[0-9A-F]{2}:){31}[0-9A-F]{2}', p['ca_fingerprint']):
        raise ValueError('Invalid proxy CA fingerprint.')
    if p['node'] != str(Path.home() / NODE_RELATIVE):
        raise ValueError('Use the reviewed side-by-side Node 24 runtime.')
    if not all(Path(p[key]).is_absolute() and Path(p[key]).is_file() for key in ('node', 'infisical', 'ca_file')):
        raise ValueError('A reviewed runtime, CLI or CA is missing.')
    if not isinstance(p['code_sha256'], dict) or set(p['code_sha256']) != source_files():
        raise ValueError('Incomplete reviewed source pins.')
    for name, digest in p['code_sha256'].items():
        target = ROOT / name
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed; trusted review is required.')
    if not isinstance(p['runtime_sha256'], dict) or set(p['runtime_sha256']) != {'node', 'infisical', 'ca_file'}:
        raise ValueError('Incomplete reviewed runtime pins.')
    for name, digest in p['runtime_sha256'].items():
        target = Path(p[name])
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Runtime or public CA changed; trusted review is required.')
    if mode == 'serve':
        verify_app_dependencies()
    return p


def sandbox_prefix(p, mode='check'):
    return ['/usr/bin/sandbox-exec', '-D', 'USER_HOME=' + str(Path.home()),
        '-D', 'REPO_ROOT=' + str(ROOT), '-D', 'CA_FILE=' + p['ca_file'],
        '-D', 'NODE_ROOT=' + str(Path(p['node']).parent.parent),
        '-D', 'NODE_BIN=' + p['node'],
        '-D', 'APP_CONFIG=' + str(ROOT / ADAPTER),
        '-f', str(ROOT / 'scripts/infisical' / ('agentbridge_app.sb' if mode == 'serve' else 'agentbridge.sb'))]


def client_command(p, mode='check'):
    if mode == 'serve':
        return sandbox_prefix(p, mode) + [p['node'], '--use-env-proxy',
            str(ROOT / 'scripts/infisical/agentbridge_app.mjs'), '--port', 'auto']
    return sandbox_prefix(p) + [p['node'], '--use-env-proxy', str(ROOT / 'scripts/infisical/agentbridge_client.mjs')]


def session_command(p, mode='check'):
    app_environment = []
    if mode == 'serve':
        app_environment = ['PATH=' + str(Path(p['node']).parent) + ':/opt/homebrew/bin:/usr/bin:/bin',
            'NEXT_TELEMETRY_DISABLED=1',
            'NODE_OPTIONS=--use-env-proxy --import=' + (ROOT / 'scripts/infisical/agentbridge_preload.mjs').as_uri()]
    return [p['infisical'], 'agent-vault', 'run', '--domain=https://app.infisical.com',
        '--project-id=5078c850-3f93-46d8-8afe-82107df8b931',
        '--access-bundle=' + p['bundle'], '--proxy=' + p['proxy'],
        '--ca-fingerprint=' + p['ca_fingerprint'], '--ttl=' + ('30m' if mode == 'serve' else '10m'), '--no-ca-trust',
        '--silent', '--telemetry=false', '--no-skills', '--', '/usr/bin/env',
        'AGENTBRIDGE_API_KEY=' + MARKER, 'AGENTBRIDGE_BASE_URL=' + p['base_url'],
        'AGENTBRIDGE_APP_CONFIG=' + str(ROOT / ADAPTER),
        'AGENTBRIDGE_DEFAULT_MODEL=' + DEFAULT_MODEL, 'NODE_ENV=development',
        'NODE_USE_ENV_PROXY=1', 'NO_PROXY=', 'no_proxy=',
        'NODE_EXTRA_CA_CERTS=' + p['ca_file'], *app_environment, *client_command(p, mode)]


def main(arguments):
    if arguments not in (['check'], ['app-check'], ['serve'], ['serve', '--port', 'auto'], ['serve', '--port=auto']):
        raise ValueError('Use check, app-check or serve --port auto; source, scope and command overrides are rejected.')
    mode = arguments[0]
    p = load_profile(mode)
    env = {'HOME': str(Path.home()), 'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'}
    if mode == 'serve':
        os.chdir(ROOT)
        os.execve(p['infisical'], session_command(p, mode), env)
    result = subprocess.run(session_command(p, mode), cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"ok":')]
    if not records:
        raise ValueError('Agent Vault check failed; verify the profile, gateway and Infisical login.')
    print(json.dumps(records[-1]))
    return 0 if result.returncode == 0 and records[-1].get('ok') is True else 1


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except ValueError as error:
        print(json.dumps({'ok': False, 'reason': str(error)}))
        sys.exit(1)
    except Exception:
        print(json.dumps({'ok': False, 'reason': 'Agent Vault check failed; diagnostic details withheld.'}))
        sys.exit(1)

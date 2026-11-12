/** Guard only the opt-in local app; hosted and default launches never import this. */
export function gatewayFetch(fetchImpl, base, validate) {
  const endpoint = new URL(base);
  return async (input, init) => {
    validate();
    const target = new URL(typeof input === 'string' || input instanceof URL ? input : input.url);
    const method = (init?.method ?? input.method ?? 'GET').toUpperCase();
    const headers = new Headers(init?.headers ?? input.headers);
    if (target.origin !== endpoint.origin || target.search || target.hash
        || target.username || target.password
        || !((method === 'GET' && target.pathname === `${endpoint.pathname}/models`)
          || (method === 'POST' && target.pathname === `${endpoint.pathname}/chat/completions`))
        || headers.get('Authorization') !== 'Bearer agentbridge_public_placeholder') {
      throw new Error('Request is outside the reviewed AgentBridge route.');
    }
    let response;
    try {
      response = await fetchImpl(input, { ...init, redirect: 'error' });
    } catch {
      throw new Error('AgentBridge connection failed; diagnostic details withheld.');
    }
    if (!response.ok) {
      await response.body?.cancel();
      return new Response(JSON.stringify({ error: { message: 'AgentBridge request failed; provider details withheld.' } }),
        { status: response.status, headers: { 'Content-Type': 'application/json' } });
    }
    return response;
  };
}

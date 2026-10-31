import {createRequire} from 'node:module';
import {registerMinimatchRegressions} from './dependency-matcher-regressions.mjs';
import {registerGlobCancellationTests} from './dependency-glob-cancellation.mjs';
const r=createRequire(import.meta.url);
const sentry=createRequire(r.resolve('@sentry/nextjs/package.json'));
const core=createRequire(sentry.resolve('@sentry/bundler-plugins/core'));
const globRequire=createRequire(core.resolve('glob'));
registerMinimatchRegressions(globRequire);
registerGlobCancellationTests(core('glob'));

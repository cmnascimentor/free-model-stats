/* FreeModelStats benchmark-v2 dashboard adapter.
 *
 * Loaded after the legacy single-file dashboard. It enriches rows from the v2
 * SQLite columns and replaces cohort-relative min/max scoring with a stable,
 * auditable cross-provider baseline score:
 *   quality 50% + transport reliability 25% + latency 15% + throughput 10%.
 * Only benchmark v2 hermes_triage rows feed the overall cross-provider score;
 * provider-specific/additional probes remain visible as evidence but cannot
 * distort the global ranking.
 */
(() => {
  'use strict';

  const CORE_PROBE = 'hermes_triage';
  const V2_PREFIX = '2.';

  const clamp = (value, min = 0, max = 100) => Math.max(min, Math.min(max, value));
  const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

  // Fixed scales: scores do not change merely because a new outlier joins the cohort.
  // Latency: 0.5s=100, 1s=80, 2s=60, 4s=40, 8s=20, 16s+=0.
  function latencyScore(ms) {
    if (!(ms > 0)) return 0;
    return clamp(100 - 20 * Math.log2(Math.max(ms, 500) / 500));
  }

  // End-to-end throughput proxy: 1=0, 2=20, 4=40, 8=60, 16=80, 32+=100 t/s.
  // This remains explicitly a proxy until streaming TTFT/generation timing is available.
  function throughputScore(tps) {
    if (!(tps > 0)) return 0;
    return clamp(20 * Math.log2(Math.max(tps, 1)));
  }

  function tableHasColumn(db, table, column) {
    try {
      const q = db.exec(`PRAGMA table_info(${table})`);
      return !!q[0]?.values?.some(row => row[1] === column);
    } catch (_) {
      return false;
    }
  }

  function enrichRunsFromDb(db, runs) {
    if (!db || !runs?.length || !tableHasColumn(db, 'model_results', 'quality_score')) return runs;
    const cols = [
      'transport_success', 'format_success', 'task_success', 'quality_score',
      'http_status', 'finish_reason', 'resolved_model', 'attempt_count',
      'final_attempt_ms', 'total_elapsed_ms', 'ttft_ms', 'generation_ms',
      'tokens_per_second', 'benchmark_version', 'probe_version'
    ];
    const q = db.exec(`SELECT run_id, model, ${cols.join(', ')} FROM model_results ORDER BY run_id ASC`);
    if (!q.length) return runs;

    const runMap = new Map(runs.map(run => [run._dbId, run]));
    for (const row of q[0].values) {
      const [runId, model, ...values] = row;
      const run = runMap.get(runId);
      if (!run) continue;
      const result = run.models.find(item => item.model === model);
      if (!result) continue;
      const data = Object.fromEntries(cols.map((name, i) => [name, values[i]]));
      result.transportSuccess = data.transport_success == null ? null : !!data.transport_success;
      result.formatSuccess = data.format_success == null ? null : !!data.format_success;
      result.taskSuccess = data.task_success == null ? !!result.success : !!data.task_success;
      result.qualityScore = data.quality_score;
      result.httpStatus = data.http_status;
      result.finishReason = data.finish_reason;
      result.resolvedModel = data.resolved_model;
      result.attemptCount = data.attempt_count;
      result.finalAttemptMs = data.final_attempt_ms;
      result.totalElapsedMs = data.total_elapsed_ms;
      result.ttftMs = data.ttft_ms;
      result.generationMs = data.generation_ms;
      result.tokensPerSecond = data.tokens_per_second;
      result.benchmarkVersion = data.benchmark_version;
      result.probeVersion = data.probe_version;
    }
    return runs;
  }

  function applyV2Scores(processed) {
    if (!processed?.runs || !processed?.modelStats) return processed;
    const { runs, modelStats, modelNames } = processed;

    for (const model of modelNames) {
      const samples = [];
      for (const run of runs) {
        if (run.probeName !== CORE_PROBE) continue;
        const result = run.models.find(item => item.model === model);
        if (!result || !String(result.benchmarkVersion || '').startsWith(V2_PREFIX)) continue;
        samples.push(result);
      }
      if (!samples.length) {
        modelStats[model].scoreVersion = 'legacy';
        continue;
      }

      const quality = mean(samples.map(r => Number(r.qualityScore ?? (r.success ? 100 : 0)))) ?? 0;
      const transportReliability = samples.filter(r => r.transportSuccess === true).length / samples.length * 100;
      const taskPassRate = samples.filter(r => r.taskSuccess === true || r.success === true).length / samples.length;
      const latencies = samples
        .filter(r => r.transportSuccess === true && Number(r.totalElapsedMs || r.responseTime) > 0)
        .map(r => Number(r.totalElapsedMs || r.responseTime));
      const throughputs = samples
        .filter(r => Number(r.tokensPerSecond) > 0)
        .map(r => Number(r.tokensPerSecond));
      const avgLatency = mean(latencies);
      const avgThroughput = mean(throughputs);
      const latency = latencyScore(avgLatency);
      const throughput = throughputScore(avgThroughput);
      const score = quality * 0.50 + transportReliability * 0.25 + latency * 0.15 + throughput * 0.10;

      const s = modelStats[model];
      s.score = Math.round(score);
      s.scoreVersion = 'v2';
      s.qualityScore = quality;
      s.transportReliability = transportReliability;
      s.latencyScore = latency;
      s.throughputScore = throughput;
      s.v2Samples = samples.length;
      s.uptime = taskPassRate; // existing UI label now reflects task-valid baseline pass rate
      s.totalRuns = samples.length;
      s.successCount = samples.filter(r => r.taskSuccess === true || r.success === true).length;
      if (avgLatency != null) s.avgTime = avgLatency;
      if (avgThroughput != null) s.avgTps = avgThroughput;
    }
    return processed;
  }

  function annotateLeaderboard() {
    try {
      const section = document.querySelector('section[data-tab="leaderboard"]');
      if (section && !section.querySelector('.benchmark-v2-note')) {
        const note = document.createElement('div');
        note.className = 'benchmark-v2-note';
        note.style.cssText = 'margin:0 0 14px;padding:10px 12px;border:1px solid var(--border);border-radius:10px;color:var(--text-dim);font-size:11.5px;background:var(--bg-card);';
        note.textContent = 'v2 overall score: quality 50% · transport reliability 25% · latency 15% · end-to-end throughput 10%. Cross-provider ranking uses only the shared hermes_triage v2 baseline.';
        const controls = section.querySelector('.table-controls');
        if (controls) controls.before(note);
      }

      document.querySelectorAll('#lb-body tr[data-model]').forEach(row => {
        const model = row.dataset.model;
        const s = typeof state !== 'undefined' ? state.modelStats?.[model] : null;
        const cell = row.querySelector('.score-cell');
        if (!cell || !s || s.scoreVersion !== 'v2') return;
        cell.title = `v2 score — quality ${s.qualityScore.toFixed(1)}, reliability ${s.transportReliability.toFixed(1)}, latency ${s.latencyScore.toFixed(1)}, throughput ${s.throughputScore.toFixed(1)}; n=${s.v2Samples}`;
        if (!cell.querySelector('.score-v2-badge')) {
          const badge = document.createElement('span');
          badge.className = 'score-v2-badge';
          badge.textContent = 'v2';
          badge.style.cssText = 'font-size:8px;padding:1px 4px;border-radius:8px;border:1px solid var(--border-bright);color:var(--text-dim);font-family:var(--font-mono);';
          cell.appendChild(badge);
        }
      });
    } catch (_) {
      // UI annotation must never break the dashboard.
    }
  }

  // Patch loaders early when this script executes before initSqlJs finishes.
  if (typeof loadFromDb === 'function') {
    const legacyLoadFromDb = loadFromDb;
    loadFromDb = function benchmarkV2LoadFromDb(db) {
      const data = legacyLoadFromDb(db);
      enrichRunsFromDb(db, data.runs);
      return data;
    };
  }

  if (typeof processData === 'function') {
    const legacyProcessData = processData;
    processData = function benchmarkV2ProcessData(data) {
      return applyV2Scores(legacyProcessData(data));
    };
  }

  if (typeof renderLbTable === 'function') {
    const legacyRenderLbTable = renderLbTable;
    renderLbTable = function benchmarkV2RenderLbTable(...args) {
      const value = legacyRenderLbTable(...args);
      annotateLeaderboard();
      return value;
    };
  }

  // Fallback for very fast cached startup where init completed before the patch
  // attached: enrich state directly, recompute, and rerender once.
  let attempts = 0;
  const timer = setInterval(() => {
    attempts += 1;
    try {
      if (typeof state !== 'undefined' && state.db && state.runs?.length && state.modelStats) {
        enrichRunsFromDb(state.db, state.runs);
        applyV2Scores({runs: state.runs, modelStats: state.modelStats, modelNames: state.modelNames || Object.keys(state.modelStats)});
        if (typeof renderLeaderboard === 'function') renderLeaderboard();
        annotateLeaderboard();
        clearInterval(timer);
      } else if (attempts >= 50) {
        clearInterval(timer);
      }
    } catch (_) {
      if (attempts >= 50) clearInterval(timer);
    }
  }, 100);
})();

import './ProgressRail.css';

function resolveStatus(done, active, previousDone) {
  if (done) return 'done';
  if (active) return 'active';
  return previousDone ? 'pending' : 'locked';
}

export default function ProgressRail({ mode, loading = {}, stage1, stage2, stage3 }) {
  const stage1Done = !!stage1;
  const stage2Done = Array.isArray(stage2) && stage2.length > 0;
  const stage3Done = !!stage3;

  const stage1Active = !!loading.stage1;
  const stage2Active = !!loading.stage2;
  const stage3Active = !!loading.stage3;

  if (mode === 'single_agent') {
    const quickStatus = resolveStatus(stage3Done, stage1Active || stage3Active, true);
    return (
      <div className="progress-rail">
        <div className="progress-heading">Live progress</div>
        <div className="progress-track single">
          <ProgressStep label="Quick response" status={quickStatus} />
          <ProgressStep label="Finalizing" status={stage3Done ? 'done' : 'active'} />
        </div>
      </div>
    );
  }

  const stage1Status = resolveStatus(stage1Done, stage1Active, true);
  const stage2Status = resolveStatus(stage2Done, stage2Active, stage1Done || stage1Active);
  const stage3Status = resolveStatus(stage3Done, stage3Active, stage2Done || stage2Active);

  return (
    <div className="progress-rail">
      <div className="progress-heading">Live progress</div>
      <div className="progress-track">
        <ProgressStep label="Stage 1 · Responses" status={stage1Status} />
        <ProgressStep label="Stage 2 · Rankings" status={stage2Status} />
        <ProgressStep label="Stage 3 · Final" status={stage3Status} />
      </div>
    </div>
  );
}

function ProgressStep({ label, status }) {
  return (
    <div className={`progress-step ${status}`}>
      <div className="step-dot" />
      <div className="step-label">{label}</div>
    </div>
  );
}

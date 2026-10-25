export function MobileGraph({ x }: { x: number }) {
  return (
    <svg
      className="diagram-mobile"
      viewBox="0 0 340 345"
      role="img"
      aria-label={`Scalar square graph: input ${x}, output ${x * x}, gradient ${2 * x}`}
    >
      <text x="20" y="24" className="svg-small">
        FORWARD ↓
      </text>
      <text x="217" y="24" className="svg-small">
        BACKWARD ↑
      </text>
      <rect x="32" y="48" width="195" height="63" rx="7" className="svg-node" />
      <text x="129" y="74" textAnchor="middle" className="svg-math">
        x = {x}
      </text>
      <text x="129" y="96" textAnchor="middle" className="svg-small">
        leaf · requires_grad
      </text>
      <path d="M129 111 V140" className="forward-line" />
      <path d="M124 134 L129 141 L134 134" className="forward-line" />
      <rect x="32" y="145" width="195" height="55" rx="7" className="svg-op" />
      <text x="129" y="178" textAnchor="middle" className="svg-math">
        square: x × x
      </text>
      <path d="M129 200 V230" className="forward-line" />
      <path d="M124 224 L129 231 L134 224" className="forward-line" />
      <rect x="32" y="235" width="195" height="63" rx="7" className="svg-node" />
      <text x="129" y="262" textAnchor="middle" className="svg-math">
        y = {x * x}
      </text>
      <text x="129" y="282" textAnchor="middle" className="svg-small">
        scalar output
      </text>
      <path d="M232 267 H279 V79 H235" className="backward-line" />
      <path d="M241 73 L234 79 L241 85" className="backward-line" />
      <text x="170" y="328" textAnchor="middle" className="svg-gradient">
        dy/dx = 2x = {2 * x}
      </text>
    </svg>
  );
}
export function MobileBatch({
  size,
  kept,
  count,
}: {
  size: number;
  kept: number;
  count: number;
}) {
  return (
    <svg
      className="diagram-mobile"
      viewBox="0 0 340 265"
      role="img"
      aria-label={`Ten samples, batch size ${size}, ${count} batches and ${10 - kept} samples dropped`}
    >
      <text x="20" y="25" className="svg-small">
        DATASET · TEN SAMPLES
      </text>
      {Array.from({ length: 10 }, (_, i) => {
        const col = i % 5,
          row = Math.floor(i / 5);
        return (
          <g key={i} opacity={i < kept ? 1 : 0.5}>
            <rect
              x={20 + col * 62}
              y={47 + row * 89}
              width="48"
              height="43"
              rx="5"
              className={i < kept ? "svg-node" : "svg-dropped"}
            />
            <text
              x={44 + col * 62}
              y={74 + row * 89}
              textAnchor="middle"
              className="svg-math"
            >
              {i}
            </text>
            <text
              x={44 + col * 62}
              y={109 + row * 89}
              textAnchor="middle"
              className="svg-small"
            >
              {i < kept ? `B${Math.floor(i / size) + 1}` : "drop"}
            </text>
          </g>
        );
      })}
      <text x="170" y="238" textAnchor="middle" className="svg-gradient">
        {count} batches · {kept} kept · {10 - kept} dropped
      </text>
    </svg>
  );
}

import type { Citation } from "../api";

interface Props {
  citations: Citation[];
}

// Renders the sources an answer was grounded in, with score + snippet so the
// user can verify every claim against the original document.
export function CitationList({ citations }: Props) {
  if (citations.length === 0) return null;
  return (
    <div className="citations">
      <div className="citations-title">Sources</div>
      <ol>
        {citations.map((c) => (
          <li key={c.marker}>
            <div className="citation-head">
              <span className="citation-marker">[{c.marker}]</span>
              <span className="citation-source">{c.source}</span>
              <span className="citation-score">score {c.score.toFixed(3)}</span>
            </div>
            <p className="citation-snippet">{c.snippet}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}

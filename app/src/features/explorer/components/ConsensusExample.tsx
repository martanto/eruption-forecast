import {
  EXAMPLE_CLASSIFIERS,
  EXAMPLE_CONSENSUS,
  EXAMPLE_INPUTS,
} from "@/features/architecture";

export function ConsensusExample() {
  return (
    <section className="calculation">
      <h3>Illustrative calculation</h3>
      <p>Invented inputs for one window; these are not run results.</p>
      {EXAMPLE_INPUTS.map((item) => (
        <p key={item.name}>
          <strong>{item.name}</strong>: probabilities [
          {item.probabilities.join(", ")}], binary votes [
          {item.votes.join(", ")}].
        </p>
      ))}
      <div className="table-scroll">
        <table>
          <caption>
            Population standard deviation; equal classifier weight
          </caption>
          <thead>
            <tr>
              <th>Group</th>
              <th>Mean P</th>
              <th>Vote</th>
              <th>Std</th>
              <th>CI-like</th>
            </tr>
          </thead>
          <tbody>
            {[
              ...EXAMPLE_CLASSIFIERS,
              { name: "Consensus", ...EXAMPLE_CONSENSUS },
            ].map((item) => (
              <tr key={item.name}>
                <th>{item.name}</th>
                <td>{item.probability.toFixed(2)}</td>
                <td>{item.vote.toFixed(2)}</td>
                <td>{item.uncertainty.toFixed(2)}</td>
                <td>{item.confidence.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Consensus mean P = (0.60 + 0.80) / 2 = 0.70.
        <br />
        Vote = (0.50 + 1.00) / 2 = 0.75.
        <br />
        CI-like = 1.96 × √(0.75 × 0.25 / 2) ≈ 0.600.
      </p>
    </section>
  );
}

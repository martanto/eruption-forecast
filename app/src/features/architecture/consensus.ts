export function aggregate(probabilities: number[], votes: number[]) {
  if (
    !probabilities.length ||
    probabilities.length !== votes.length ||
    [...probabilities, ...votes].some(
      (value) => !Number.isFinite(value) || value < 0 || value > 1,
    )
  )
    throw new Error(
      "Expected aligned, nonempty probability and vote arrays in [0, 1].",
    );
  const mean = (values: number[]) =>
    values.reduce((sum, value) => sum + value, 0) / values.length;
  const probability = mean(probabilities);
  const vote = mean(votes);
  return {
    probability,
    vote,
    uncertainty: Math.sqrt(
      mean(probabilities.map((value) => (value - probability) ** 2)),
    ),
    confidence: 1.96 * Math.sqrt((vote * (1 - vote)) / votes.length),
  };
}
export const EXAMPLE_INPUTS = [
  { name: "RF", probabilities: [0.8, 0.4], votes: [1, 0] },
  { name: "Lite-RF", probabilities: [0.7, 0.9], votes: [1, 1] },
];
export const EXAMPLE_CLASSIFIERS = EXAMPLE_INPUTS.map((item) => ({
  name: item.name,
  ...aggregate(item.probabilities, item.votes),
}));
export const EXAMPLE_CONSENSUS = aggregate(
  EXAMPLE_CLASSIFIERS.map((item) => item.probability),
  EXAMPLE_CLASSIFIERS.map((item) => item.vote),
);

const DecisionTree = require("decision-tree");

// Dataset with features and labels
const trainingData = [
  { text: "FDA approves medication XYZ for diabetes.", cure: 0, label: 1 },
  { text: "Cure diabetes in 7 days! Click here to order.", cure: 1, label: 0 },
  { text: "Visit CDC for accurate health info.", cure: 0, label: 1 },
  {
    text: "Dr. John Doe says: 'This is the ultimate cure!'",
    cure: 1,
    label: 0,
  },
  {
    text: "Journal reports lifestyle changes reduce risks.",
    cure: 0,
    label: 1,
  },
  {
    text: "Congratulations! You’ve been selected for a miracle cure!",
    cure: 1,
    label: 0,
  },
  {
    text: "Apply for tax relief at irs.gov.",
    cure: 0,
    label: 1,
  },
  {
    text: "Win a FREE trip to Bali! Limited slots available!!!",
    cure: 1,
    label: 0,
  },
  {
    text: "WHO updates COVID-19 guidelines. Visit who.int for details.",
    cure: 0,
    label: 1,
  },
  {
    text: "Earn $500/day from home! Click here!!!",
    cure: 1,
    label: 0,
  },
];

// Features and target label
const features = ["cure"];
const className = "label";

// Training the Decision Tree
const dt = new DecisionTree(trainingData, className, features);

// Testing the model with 10 examples
const testData = [
  { text: "FDA approves a new treatment.", cure: 0 },
  { text: "Free cure for diabetes now available!", cure: 1 },
  { text: "WHO recommends booster shots for better immunity.", cure: 0 },
  { text: "Win a FREE car! Sign up today!!!", cure: 1 },
  { text: "Government updates visa policies. Visit usa.gov.", cure: 0 },
  { text: "Claim your $10,000 prize now! Limited offer!", cure: 1 },
  { text: "CDC announces new guidelines for flu prevention.", cure: 0 },
  { text: "Lose weight instantly! Order your miracle pill today.", cure: 1 },
  { text: "NASA reveals new findings on Mars exploration.", cure: 0 },
  { text: "Buy one get one FREE! Act now!!!", cure: 1 },
];

// Predicting results
testData.forEach((test) => {
  const prediction = dt.predict(test);
  console.log(`Text: "${test.text}" => Predicted Class: ${prediction}`);
});

// Calculating accuracy on the training data
const accuracy = dt.evaluate(trainingData);
console.log(`Accuracy: ${accuracy * 100}%`);

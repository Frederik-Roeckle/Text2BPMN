const { convertAll } = require("bpmn-to-image");

(async () => {
  await convertAll([{ input: process.argv[2], outputs: [process.argv[3]] }]);
})();

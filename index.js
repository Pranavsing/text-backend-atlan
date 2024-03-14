const express = require("express");
const app = express();
const port = 4000;

const {models,featuredModels} = require("./data")
const cors = require("cors");
app.get("/", (req, res) => {
	res.send("Hello There");
});
app.use(cors())
app.get("/models", (req, res) => {
	res.send(JSON.stringify(models));
});
app.get("/featuredModels", (req, res) => {
	res.send(featuredModels);
});

app.listen(port, () => {
	console.log(`Server Running on port ${port}`);
});

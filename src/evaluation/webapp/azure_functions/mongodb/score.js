const mongoose = require("mongoose")

const scoreSchema = new mongoose.Schema({
    textId: String,
    scorerId: String,
    appropriateness: Number,
    fluency: Number,
    diversity: Number,
    overall: Number
});

mongoose.model("Score", scoreSchema, "Score");

module.exports = {
    scoreSchema
}
const mongoose = require("mongoose")

const scorerSchema = new mongoose.Schema({
    name: String,
    scorerId: String
});

mongoose.model("Scorer", scorerSchema, "Scorer");

module.exports = {
    scorerSchema
}
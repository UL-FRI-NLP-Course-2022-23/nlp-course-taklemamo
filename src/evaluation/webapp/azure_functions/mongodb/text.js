const mongoose = require("mongoose")

const textSchema = new mongoose.Schema({
    originalText: String,
    paraphrasedText: String
});

mongoose.model("Text", textSchema, "Text");

module.exports = {
    textSchema
};
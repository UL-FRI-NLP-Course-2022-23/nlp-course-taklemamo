const mongoose = require("mongoose")

const textSchema = new mongoose.Schema({
    textId: String,
    originalText: String,
    paraphrasedText: String
});

mongoose.model("Text", textSchema, "Text");

module.exports = {
    textSchema
};
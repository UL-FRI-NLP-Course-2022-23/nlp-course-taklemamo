const mongoose = require("mongoose")
const db = require("../mongodb/db")

const scoreSchema = require("../mongodb/score").scoreSchema
const textSchema = require("../mongodb/text").textSchema

const Score = mongoose.model("Score", scoreSchema)
const Text = mongoose.model("Text", textSchema)

const getScores = async (context) => {
    db.connectToDatabase();

    await Score.find().then((scores) => {
        context.res = {
            status: 200,
            body: scores
        };
    }).catch((err) => { 
        context.res = {
            status: 400,
            body: "Error getting scores." + err
        };
    });
}

const addScore = async (context, score) => {
    db.connectToDatabase();

    const newScore = new Score(score)
    await newScore.save().then((score) => {
        context.res = {
            status: 201,
            body: score
        };
    }).catch((err) => {
        context.res = {
            status: 400,
            body: "Error adding score." + err
        };
    });
}

module.exports = {
    getScores,
    addScore
}
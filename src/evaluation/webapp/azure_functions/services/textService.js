const mongoose = require("mongoose")
const db = require("../mongodb/db")

const textSchema = require("../mongodb/text").textSchema

const Text = mongoose.model("Text", textSchema)

const getTexts = async (context) => {
    db.connectToDatabase();

    await Text.find().then((texts) => {
        context.res = {
            status: 200,
            body: texts
        };
    }).catch((err) => { 
        context.res = {
            status: 400,
            body: "Error getting texts." + err
        };
    });
}

const getText = async (context, id) => {
    db.connectToDatabase();

    await Text.find({textId: id}).then((text) => {
        if (text.length == 0) {
            context.res = {
                status: 404,
                body: "Text not found.",
                headers: { "Access-Control-Allow-Origin": "*" }
            };
        }
        else {
            context.res = {
                status: 200,
                body: text,
                headers: { "Access-Control-Allow-Origin": "*" }
            };
        }
    }).catch((err) => { 
        context.res = {
            status: 400,
            body: "Error getting text." + err,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    });
}

const addText = async (context, text) => {
    db.connectToDatabase();

    const newScore = new Text(text);
    await newScore.save().then((text) => {
        context.res = {
            status: 201,
            body: text
        };
    }).catch((err) => { 
        context.res = {
            status: 400,
            body: "Error adding text." + err
        };
    });
}

module.exports = {
    getTexts,
    getText,
    addText
}
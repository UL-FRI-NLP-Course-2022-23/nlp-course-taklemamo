const mongoose = require("mongoose")

var dbURI = "mongodb+srv://taklemamo:YS8tODUJ3HLt8LSc@nlpproj.lct9rsv.mongodb.net/?retryWrites=true&w=majority"

let db=null

const connectToDatabase = async () => {
    if (db) 
        return db

    db = await mongoose.connect(dbURI, { useNewUrlParser: true, useUnifiedTopology: true })

    mongoose.connection.on('connected', function () {
        console.log('Mongoose connected to ' + dbURI)
    });
    
    mongoose.connection.on("error", (error) => {
        console.log("Mongoose error connecting: ", error)
    });
    
    mongoose.connection.on("disconnected", () => {
        console.log("Mongoose is disconnected.")
    });

    return db
}

require("./text")
require("./score")

module.exports = {
    connectToDatabase
}
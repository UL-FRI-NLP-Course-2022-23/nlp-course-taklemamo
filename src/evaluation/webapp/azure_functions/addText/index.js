const textService = require("../services/textService")

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');

    if(req.body) {
        await textService.addText(context, req.body);
    }
    else {
        context.res = {
            status: 400,
            body: "Text not added. No body in request."
        };
    }
}
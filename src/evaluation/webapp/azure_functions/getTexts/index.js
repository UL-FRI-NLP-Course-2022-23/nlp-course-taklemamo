const textService = require('../services/textService');

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');

    await textService.getTexts(context);
}
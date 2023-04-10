const todoService = require('../services/todo');

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');

    todoService.getTodos(context);
};
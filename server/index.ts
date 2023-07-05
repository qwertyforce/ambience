import config from './../config/config'
import fastifyMultipart from '@fastify/multipart'
import fastify from 'fastify'
import fastifyReplyFrom from '@fastify/reply-from'
import fastifyCors from '@fastify/cors'
// import busboy from '@fastify/busboy'
// import fs from 'fs'
// import path from 'path'
const server = fastify({logger:true})

// server.addHook('onRequest', function (req, _reply, done) {
//     console.log(req.raw)
//     if (req.raw.headers['content-type'] && req.raw.headers['content-type'].includes("multipart/form-data;")) {
//         const bb = busboy({
//             headers: { "content-type": req.raw.headers['content-type'] }, limits: {
//                 fieldNameSize: 100, // Max field name size in bytes
//                 fieldSize: 10,     // Max field value size in bytes
//                 fields: 10,         // Max number of non-file fields
//                 fileSize: 50000000,  // For multipart forms, the max file size in bytes  //50MB
//                 files: 1,           // Max number of file fields
//                 headerPairs: 2000   // Max number of header key=>value pairs
//             }
//         })
//         bb.on('file', (_name, file, _info) => {
//             const saveTo = path.join("./uploads", `${new Date().getTime()}.jpg`)
//             file.pipe(fs.createWriteStream(saveTo))
//         });
//         req.raw.pipe(bb)
//     }
//     done()
//   })

server.register(fastifyCors, {
    "origin": "*",
    "methods": "GET,HEAD,PUT,PATCH,POST,DELETE",
  })
const port = config.server_port

function combineURLs(baseURL: string, relativeURL: string) { //https://stackoverflow.com/a/49966753
    return relativeURL
        ? baseURL.replace(/\/+$/, '') + '/' + relativeURL.replace(/^\/+/, '')
        : baseURL;
}

import calculate_all_image_features from "./routes/calculate_all_image_features"
import delete_all_image_features from "./routes/delete_all_image_features"
import reverse_search from "./routes/reverse_search"

server.register(async function (app) {
    app.register(fastifyMultipart, {
        attachFieldsToBody: true,
        limits: {
            fieldNameSize: 100, // Max field name size in bytes
            fieldSize: 10,     // Max field value size in bytes
            fields: 10,         // Max number of non-file fields
            fileSize: 50000000,  // For multipart forms, the max file size in bytes  //50MB
            files: 1,           // Max number of file fields
            headerPairs: 2000   // Max number of header key=>value pairs
        }
    })
    app.post("/calculate_all_image_features", calculate_all_image_features)
    app.post("/delete_all_image_features", delete_all_image_features)
    app.post("/reverse_search", reverse_search)
})

///////////////////////////////////////////////////////////////////////////////////////////////PROXY
server.register(fastifyReplyFrom,{http: {agentOptions: {keepAliveMsecs: 10 * 60 * 1000},requestOptions: {timeout: 10 * 60 * 1000 }}})
server.addContentTypeParser('multipart/form-data', function (_request, payload, done) {
    done(null, payload)  //https://github.com/fastify/help/issues/67
})

const local_features_routes = ['/local_features_get_similar_images_by_image_buffer','/local_features_get_similar_images_by_id', '/calculate_local_features', '/delete_local_features']
local_features_routes.forEach((r) => server.post(r, (_req, res) => {
    try {
        res.from(combineURLs(config.local_features_microservice_url, r))
    } catch (err) {
        console.log(err)
        res.status(500).send('Local features microservice is down')
    }
}))

const global_features_routes = ['/global_features_get_similar_images_by_image_buffer', '/global_features_get_similar_images_by_id', '/calculate_global_features', '/delete_global_features']
global_features_routes.forEach((r) => server.post(r, (_req, res) => {
    try {
        res.from(combineURLs(config.global_features_microservice_url, r))
    } catch (err) {
        console.log(err)
        res.status(500).send('Global features microservice is down')
    }
}))

const image_text_features_routes = ['/image_text_features_get_similar_images_by_image_buffer', '/image_text_features_get_similar_images_by_id', '/image_text_features_get_similar_images_by_text', '/calculate_image_text_features', '/delete_image_text_features']
image_text_features_routes.forEach((r) => server.post(r, (_req, res) => {
    try {
        res.from(combineURLs(config.image_text_features_microservice_url, r))
    } catch (err) {
        console.log(err)
        res.status(500).send('image_text features microservice is down')
    }
}))

const color_routes = ['/color_get_similar_images_by_image_buffer', '/color_get_similar_images_by_id', '/calculate_color_features', '/delete_color_features']
color_routes.forEach((r) => server.post(r, (_req, res) => {
    try {
        res.from(combineURLs(config.color_microservice_url, r))
    } catch (err) {
        console.log(err)
        res.status(500).send('Color features microservice is down')
    }
}))

// const text_routes = ['/text_get_similar_images_by_image_buffer', '/text_get_similar_images_by_id', '/calculate_text_features', '/delete_text_features']
// text_routes.forEach((r) => server.post(r, (_req, res) => {
//     try {
//         res.from(combineURLs(config.text_microservice_url, r))
//     } catch (err) {
//         console.log(err)
//         res.status(500).send('Text features microservice is down')
//     }
// }))

const phash_routes = ['/phash_get_similar_images_by_image_buffer', '/calculate_phash_features', '/delete_phash_features']
phash_routes.forEach((r) => server.post(r, (_req, res) => {
    try {
        res.from(combineURLs(config.phash_microservice_url, r))
    } catch (err) {
        console.log(err)
        res.status(500).send('Phash microservice is down')
    }
}))

server.post("/get_image_tags", (_req, res) => {
    try {
        res.from(combineURLs(config.image_tags_microservice_url, "/get_image_tags"))
    } catch (err) {
        console.log(err)
        res.status(500).send('image tagging microservice is down')
    }
})

server.post("/get_image_caption", (_req, res) => {
    try {
        res.from(combineURLs(config.image_caption_microservice_url, "/get_image_caption"))
    } catch (err) {
        console.log(err)
        res.status(500).send('image caption microservice is down')
    }
})
////////////////////////////////////////////////////////////////////////////////////////////////////////////

server.listen({port:port, host:"0.0.0.0"}, function (err, address) {
    if (err) {
        console.error(err)
        process.exit(1)
    }
    console.log(`server listening on ${address}`)
})

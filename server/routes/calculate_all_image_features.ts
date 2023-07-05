import  { FastifyRequest, FastifyReply } from "fastify"
import { FromSchema } from "json-schema-to-ts";
import image_ops from "./../helpers/image_ops"
const body_schema_calculate_all_image_features = {
    type: 'object',
    properties: {
        image: {
            type: 'object',
            properties: {
                encoding: { type: 'string' },
                filename: { type: 'string' },
                limit: { type: 'boolean' },
                mimetype: { type: 'string' }
            }
        },
        name: { type: 'string' },
        image_id: {
            type: "object",
            properties: {
                value: { type: 'string' }
            }
        },
    },
    required: ['image', 'image_id'],
} as const;

async function calculate_all_image_features(req: FastifyRequest<{ Body: FromSchema<typeof body_schema_calculate_all_image_features> }>, res: FastifyReply) {
    let image_buffer: Buffer;
    try {
        image_buffer = await (req as any).body.image.toBuffer()
    } catch (err) {
        return res.status(500).send()
    }
    if (req.body.image_id.value) {
        const image_id = parseInt(req.body.image_id.value)
        const results:any = await image_ops.calculate_all_image_features(image_id, image_buffer)
        console.log(results[3])
        for(let i=0;i<results.length;i++){
            const new_obj:any = {status:results[i].status}
            if(results[i]?.reason?.message){
                new_obj.reason = results[i].reason.message
            }
            if(results[i]?.value!==undefined){
                new_obj.value=results[i].value
            }
            if(results[i]?.reason?.response?.data){
                new_obj.data=results[i]?.reason?.response?.data
            }
            results[i] = new_obj
        }
        console.log(results)
        res.send(results)
    }
}

export default {
    schema: {
        body: body_schema_calculate_all_image_features
    },
    handler: calculate_all_image_features
}
import { FastifyRequest, FastifyReply } from "fastify"
import { FromSchema } from "json-schema-to-ts";
import image_ops from "./../helpers/image_ops"
const body_schema_delete_all_image_features = {
    type: 'object',
    properties: {
        image_id: { type: "number" }
    },
    required: ['image_id'],
} as const;

async function delete_all_image_features(req: FastifyRequest<{ Body: FromSchema<typeof body_schema_delete_all_image_features> }>, res: FastifyReply) {
    const results:any = await image_ops.delete_all_image_features(req.body.image_id)
    for(let i=0;i<results.length;i++){
        const new_obj:any = {status:results[i].status}
        if(results[i]?.reason?.message){
            new_obj.reason = results[i].reason.message
        }
        if(results[i]?.reason?.response?.data){
            new_obj.data=results[i]?.reason?.response?.data
        }
        if(results[i]?.reason?.config?.url){
            new_obj.url=results[i]?.reason?.config?.url
        }
        if(results[i]?.value){
            new_obj.value=results[i].value.data
            new_obj.url=results[i].value.config.url
        }
        results[i] = new_obj
    }
    console.log(results)
    res.send(results)
}

export default {
    schema: {
        body: body_schema_delete_all_image_features
    },
    handler: delete_all_image_features
}
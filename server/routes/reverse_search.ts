import { FastifyRequest, FastifyReply } from "fastify"
import { FromSchema } from "json-schema-to-ts";
import image_ops from "./../helpers/image_ops"
const body_schema_reverse_search = {
    type: 'object',
    properties: {
        image: { $ref: '#mySharedSchema' },
    },
    required: ['image'],
} as const;

async function reverse_search(req: FastifyRequest<{ Body: FromSchema<typeof body_schema_reverse_search> }>, res: FastifyReply) {
    let image_buffer: Buffer;
    try {
        image_buffer = await (req as any).body.image.toBuffer()
    } catch (err) {
        return res.status(500).send()
    }
    const results = await image_ops.get_similar_images(image_buffer)
    return results
}

export default {
    schema: {
        body: body_schema_reverse_search
    },
    handler: reverse_search
}
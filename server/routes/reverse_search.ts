import { FastifyRequest, FastifyReply } from "fastify"
import { FromSchema } from "json-schema-to-ts"
import image_ops from "./../helpers/image_ops"
const body_schema_reverse_search = {
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
        find_duplicate: {
            type: 'object',
            properties: {
                fieildname: { type: 'string' },
                encoding: { type: 'string' },
                value: { type: 'string' }
            }
        },
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
    const results = await image_ops.get_similar_images(image_buffer, Boolean(parseInt(req.body?.find_duplicate?.value || "0")))
    return results
}

export default {
    schema: {
        body: body_schema_reverse_search
    },
    handler: reverse_search
}
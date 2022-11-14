import torch
import sys

uni_state = sys.argv[1]
uni_state_dict = torch.load(uni_state)['ema']['params']
new_model={}
for k, v in uni_state_dict.items():
    new_k = k[6:]
    if 'input_embedder.linear_tf_m' in k:
        new_k = new_k.replace('input_embedder.linear_tf_m', 'InputEmbedder.preprocessing_1d')
        new_model[new_k] = v
        continue
    if 'input_embedder.linear_msa_m' in k:
        new_k = new_k.replace('input_embedder.linear_msa_m', 'InputEmbedder.preprocess_msa')
        new_model[new_k] = v
        continue
    if 'input_embedder.linear_tf_z_i' in k:
        new_k = new_k.replace('input_embedder.linear_tf_z_i','InputEmbedder.left_single')
        new_model[new_k] = v
        continue
    if 'input_embedder.linear_tf_z_j' in k:
        new_k = new_k.replace('input_embedder.linear_tf_z_j','InputEmbedder.right_single')
        new_model[new_k] = v
        continue
    if 'recycling_embedder.linear' in k:
        new_k = new_k.replace('recycling_embedder.linear', 'InputEmbedder.RecyclingEmbedder.prev_pos_linear')
        new_model[new_k] = v
        continue
    if 'recycling_embedder.layer_norm_m' in k:
        new_k = new_k.replace('recycling_embedder.layer_norm_m','InputEmbedder.RecyclingEmbedder.prev_msa_first_row_norm')
        new_model[new_k] = v
        continue
    if 'recycling_embedder.layer_norm_z' in k:
        new_k = new_k.replace('recycling_embedder.layer_norm_z','InputEmbedder.RecyclingEmbedder.prev_pair_norm')
        new_model[new_k] = v
        continue
    if 'template_angle_embedder.linear_1' in k:
        new_k = new_k.replace('template_angle_embedder.linear_1','TemplateEmbedding1D.template_single_embedding')
        new_model[new_k] = v
        continue
    if 'template_angle_embedder.linear_2' in k:
        new_k = new_k.replace('template_angle_embedder.linear_2','TemplateEmbedding1D.template_projection')
        new_model[new_k] = v
        continue
    for i in range(8):
        tmp = 'template_pair_embedder.linear.' + str(i)
        rep_tmp = 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_'+str(i)
        if tmp in k:
            new_k = new_k.replace(tmp, rep_tmp)
            new_model[new_k] = v
            continue
    if 'template_pair_embedder.z_linear' in k:
        new_k = new_k.replace('template_pair_embedder.z_linear', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_8')
        new_model[new_k] = v
        continue
    if 'template_pair_embedder.z_layer_norm' in k:
        new_k = new_k.replace('template_pair_embedder.z_layer_norm','InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.query_embedding_norm')
        new_model[new_k] = v
        continue
    if 'template_pair_stack.blocks.' in k:
        for i in range(2):
            if('template_pair_stack.blocks.'+str(i)+'.tri_att_start.layer_norm' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_att_start.layer_norm', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleAttentionStartingNode.query_norm')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_att_start.linear' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_att_start.linear', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleAttentionStartingNode.feat_2d_weights')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_att_start.mha' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_att_start.mha', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleAttentionStartingNode.mha')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_att_end.layer_norm' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_att_end.layer_norm', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleAttentionEndingNode.query_norm')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_att_end.linear' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_att_end.linear', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleAttentionEndingNode.feat_2d_weights')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_att_end.mha' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_att_end.mha', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleAttentionEndingNode.mha')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_g' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_g', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.gating_linear')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_z' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_z', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.output_projection')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.layer_norm_in' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.layer_norm_in', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.layer_norm_input')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.layer_norm_out' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.layer_norm_out', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.center_layer_norm')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_ab_p.weight' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.left_projection.weight'] = v[:idx, :]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.right_projection.weight'] = v[idx:, :]
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_ab_p.bias' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.left_projection.bias'] = v[:idx]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.right_projection.bias'] = v[idx:]
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_ab_g.weight' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.left_gate.weight'] = v[:idx, :]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.right_gate.weight'] = v[idx:, :]
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_out.linear_ab_g.bias' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.left_gate.bias'] = v[:idx]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationOutgoing.right_gate.bias'] = v[idx:]

            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_g' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_g', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.gating_linear')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_z' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_z', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.output_projection')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.layer_norm_in' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.layer_norm_in', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.layer_norm_input')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.layer_norm_out' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.layer_norm_out', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.center_layer_norm')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_ab_p.weight' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.left_projection.weight'] = v[idx:, :]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.right_projection.weight'] = v[:idx, :]
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_ab_p.bias' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.left_projection.bias'] = v[idx:]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.right_projection.bias'] = v[:idx]
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_ab_g.weight' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.left_gate.weight'] = v[idx:, :]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.right_gate.weight'] = v[:idx, :]
            if('template_pair_stack.blocks.'+str(i)+'.tri_mul_in.linear_ab_g.bias' in k):
                idx = v.shape[0]//2
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.left_gate.bias'] = v[idx:]
                new_model['InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.TriangleMultiplicationIngoing.right_gate.bias'] = v[:idx]
            if('template_pair_stack.blocks.'+str(i)+'.pair_transition.layer_norm' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.pair_transition.layer_norm', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.PairTransition.input_layer_norm')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.pair_transition.linear_1' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.pair_transition.linear_1', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.PairTransition.transition1')
                new_model[new_k] = v
            if('template_pair_stack.blocks.'+str(i)+'.pair_transition.linear_2' in k):
                new_k = new_k.replace('template_pair_stack.blocks.'+str(i)+'.pair_transition.linear_2', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration.'+str(i)+'.PairTransition.transition2')
                new_model[new_k] = v
    if('template_pair_stack.layer_norm' in k):
        new_k = new_k.replace('template_pair_stack.layer_norm', 'InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.output_layer_norm')
        new_model[new_k] = v
        continue
    if('template_proj.output_linear' in k):
        new_k = new_k.replace('template_proj.output_linear', 'InputEmbedder.TemplateEmbedding.output_linear')
        new_model[new_k] = v
        continue
    if('extra_msa_embedder.linear' in k):
        new_k = new_k.replace('extra_msa_embedder.linear', 'InputEmbedder.extra_msa_activations')
        new_model[new_k] = v
        continue
    for i in range(4):
        if('extra_msa_stack.blocks.'+str(i)+'.msa_att_row.layer_norm_m' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.'+str(i)+'.msa_att_row.layer_norm_m', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.RowAttentionWithPairBias.query_norm')
            new_model[new_k] = v
        if('extra_msa_stack.blocks.'+str(i)+'.msa_att_row.layer_norm_z' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.'+str(i)+'.msa_att_row.layer_norm_z', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.RowAttentionWithPairBias.feat_2d_norm')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_row.linear_z' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_row.linear_z',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.RowAttentionWithPairBias.feat_2d_weights')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_row.mha' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_row.mha',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.RowAttentionWithPairBias.mha')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.layer_norm_m' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.layer_norm_m', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.ExtraColumnGlobalAttention.query_norm')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_q' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_q', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.ExtraColumnGlobalAttention.q')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_k' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_k', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.ExtraColumnGlobalAttention.k')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_v' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_v', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.ExtraColumnGlobalAttention.v')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_g' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_g', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.ExtraColumnGlobalAttention.gate')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_o' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_att_col.global_attention.linear_o', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.ExtraColumnGlobalAttention.output')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_transition.layer_norm' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_transition.layer_norm', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.RecTransition.input_layer_norm')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_transition.linear_1' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_transition.linear_1', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.RecTransition.transition1')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.msa_transition.linear_2' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.msa_transition.linear_2', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.RecTransition.transition2')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.layer_norm.' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.layer_norm.', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.OuterProductMean.layer_norm_input.')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.linear_1' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.linear_1', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.OuterProductMean.left_projection')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.linear_2' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.linear_2', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.OuterProductMean.right_projection')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.linear_out' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.outer_product_mean.linear_out', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.OuterProductMean.output')
            new_model[new_k] = v
        if('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_g' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_g', 'InputEmbedder.FragExtraStack.layers.'+str(i)+'.TriangleMultiplicationOutgoing.gating_linear')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_z' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_z',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.output_projection')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.layer_norm_in' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.layer_norm_in',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.layer_norm_input')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.layer_norm_out' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.layer_norm_out',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.center_layer_norm')
            new_model[new_k] = v

        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_g' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_g',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.gating_linear')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_z' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_z',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.output_projection')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.layer_norm_in' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.layer_norm_in',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.layer_norm_input')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.layer_norm_out' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.layer_norm_out',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.center_layer_norm')
            new_model[new_k] = v

        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_ab_p.weight' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.left_projection.weight'] = v[idx:,:]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.right_projection.weight'] = v[:idx,:]
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_ab_p.bias' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.left_projection.bias'] = v[idx:]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.right_projection.bias'] = v[:idx]
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_ab_g.weight' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.left_gate.weight'] = v[idx:,:]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.right_gate.weight'] = v[:idx,:]
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_in.linear_ab_g.bias' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.left_gate.bias'] = v[idx:]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationIngoing.right_gate.bias'] = v[:idx]

        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_ab_p.weight' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.left_projection.weight'] = v[:idx,:]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.right_projection.weight'] = v[idx:,:]
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_ab_p.bias' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.left_projection.bias'] = v[:idx]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.right_projection.bias'] = v[idx:]
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_ab_g.weight' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.left_gate.weight'] = v[:idx,:]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.right_gate.weight'] = v[idx:,:]
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_mul_out.linear_ab_g.bias' in k):
            idx = v.shape[0] // 2
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.left_gate.bias'] = v[:idx]
            new_model[
                'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleMultiplicationOutgoing.right_gate.bias'] = v[idx:]

        if ('extra_msa_stack.blocks.' + str(i) + '.tri_att_start.layer_norm' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_att_start.layer_norm',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleAttentionStartingNode.query_norm')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_att_start.linear' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_att_start.linear',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleAttentionStartingNode.feat_2d_weights')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_att_start.mha' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_att_start.mha',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleAttentionStartingNode.mha')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_att_end.layer_norm' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_att_end.layer_norm',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleAttentionEndingNode.query_norm')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_att_end.linear' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_att_end.linear',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleAttentionEndingNode.feat_2d_weights')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.tri_att_end.mha' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.tri_att_end.mha',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.TriangleAttentionEndingNode.mha')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.pair_transition.layer_norm' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.pair_transition.layer_norm',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.PairTransition.input_layer_norm')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.pair_transition.linear_1' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.pair_transition.linear_1',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.PairTransition.transition1')
            new_model[new_k] = v
        if ('extra_msa_stack.blocks.' + str(i) + '.pair_transition.linear_2' in k):
            new_k = new_k.replace('extra_msa_stack.blocks.' + str(i) + '.pair_transition.linear_2',
                                  'InputEmbedder.FragExtraStack.layers.' + str(i) + '.PairTransition.transition2')
            new_model[new_k] = v
    if ('evoformer.blocks.' in k):
        for i in range(48):
            if('evoformer.blocks.'+str(i)+'.msa_att_row.layer_norm_m' in k):
                new_k = new_k.replace('evoformer.blocks.'+str(i)+'.msa_att_row.layer_norm_m', 'Evoformer.'+str(i)+'.RowAttentionWithPairBias.query_norm')
                new_model[new_k] = v
            if('evoformer.blocks.'+str(i)+'.msa_att_row.layer_norm_z' in k):
                new_k = new_k.replace('evoformer.blocks.'+str(i)+'.msa_att_row.layer_norm_z', 'Evoformer.'+str(i)+'.RowAttentionWithPairBias.feat_2d_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_att_row.linear_z' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_att_row.linear_z',
                                  'Evoformer.' + str(i) + '.RowAttentionWithPairBias.feat_2d_weights')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_att_row.mha' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_att_row.mha',
                                  'Evoformer.' + str(i) + '.RowAttentionWithPairBias.mha')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_att_col.layer_norm_m' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_att_col.layer_norm_m', 'Evoformer.'+str(i)+'.LigColumnAttention.query_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_att_col.mha' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_att_col.mha', 'Evoformer.'+str(i)+'.LigColumnAttention.mha')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_transition.layer_norm' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_transition.layer_norm', 'Evoformer.'+str(i)+'.RecTransition.input_layer_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_transition.linear_1' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_transition.linear_1', 'Evoformer.'+str(i)+'.RecTransition.transition1')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.msa_transition.linear_2' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.msa_transition.linear_2', 'Evoformer.'+str(i)+'.RecTransition.transition2')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.outer_product_mean.layer_norm.' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.outer_product_mean.layer_norm.', 'Evoformer.'+str(i)+'.OuterProductMean.layer_norm_input.')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.outer_product_mean.linear_1' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.outer_product_mean.linear_1', 'Evoformer.'+str(i)+'.OuterProductMean.left_projection')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.outer_product_mean.linear_2' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.outer_product_mean.linear_2', 'Evoformer.'+str(i)+'.OuterProductMean.right_projection')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.outer_product_mean.linear_out' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.outer_product_mean.linear_out', 'Evoformer.'+str(i)+'.OuterProductMean.output')
                new_model[new_k] = v
            if('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_g' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_g', 'Evoformer.'+str(i)+'.TriangleMultiplicationOutgoing.gating_linear')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_z' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_z',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.output_projection')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.layer_norm_in' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_out.layer_norm_in',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.layer_norm_input')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.layer_norm_out' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_out.layer_norm_out',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.center_layer_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_g' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_g',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.gating_linear')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_z' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_z',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.output_projection')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.layer_norm_in' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_in.layer_norm_in',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.layer_norm_input')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.layer_norm_out' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_mul_in.layer_norm_out',
                                      'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.center_layer_norm')
                new_model[new_k] = v

            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_ab_p.weight' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.left_projection.weight'] = v[idx:,:]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.right_projection.weight'] = v[:idx,:]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_ab_p.bias' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.left_projection.bias'] = v[idx:]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.right_projection.bias'] = v[:idx]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_ab_g.weight' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.left_gate.weight'] = v[idx:,:]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.right_gate.weight'] = v[:idx,:]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_in.linear_ab_g.bias' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.left_gate.bias'] = v[idx:]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationIngoing.right_gate.bias'] = v[:idx]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_ab_p.weight' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.left_projection.weight'] = v[:idx,:]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.right_projection.weight'] = v[idx:,:]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_ab_p.bias' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.left_projection.bias'] = v[:idx]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.right_projection.bias'] = v[idx:]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_ab_g.weight' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.left_gate.weight'] = v[:idx,:]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.right_gate.weight'] = v[idx:,:]
            if ('evoformer.blocks.' + str(i) + '.tri_mul_out.linear_ab_g.bias' in k):
                idx = v.shape[0] // 2
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.left_gate.bias'] = v[:idx]
                new_model[
                    'Evoformer.' + str(i) + '.TriangleMultiplicationOutgoing.right_gate.bias'] = v[idx:]
            if ('evoformer.blocks.' + str(i) + '.tri_att_start.layer_norm' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_att_start.layer_norm',
                                      'Evoformer.' + str(i) + '.TriangleAttentionStartingNode.query_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_att_start.linear' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_att_start.linear',
                                      'Evoformer.' + str(i) + '.TriangleAttentionStartingNode.feat_2d_weights')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_att_start.mha' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_att_start.mha',
                                      'Evoformer.' + str(i) + '.TriangleAttentionStartingNode.mha')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_att_end.layer_norm' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_att_end.layer_norm',
                                      'Evoformer.' + str(i) + '.TriangleAttentionEndingNode.query_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_att_end.linear' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_att_end.linear',
                                      'Evoformer.' + str(i) + '.TriangleAttentionEndingNode.feat_2d_weights')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.tri_att_end.mha' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.tri_att_end.mha',
                                      'Evoformer.' + str(i) + '.TriangleAttentionEndingNode.mha')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.pair_transition.layer_norm' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.pair_transition.layer_norm',
                                      'Evoformer.' + str(i) + '.PairTransition.input_layer_norm')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.pair_transition.linear_1' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.pair_transition.linear_1',
                                      'Evoformer.' + str(i) + '.PairTransition.transition1')
                new_model[new_k] = v
            if ('evoformer.blocks.' + str(i) + '.pair_transition.linear_2' in k):
                new_k = new_k.replace('evoformer.blocks.' + str(i) + '.pair_transition.linear_2',
                                      'Evoformer.' + str(i) + '.PairTransition.transition2')
                new_model[new_k] = v
    if ('evoformer.linear' in k):
        new_k = new_k.replace('evoformer.linear', 'EvoformerExtractSingleRec')
        new_model[new_k] = v
        continue
    if ('structure_module.layer_norm_s' in k):
        new_k = new_k.replace('structure_module.layer_norm_s', 'StructureModule.single_layer_norm')
        new_model[new_k] = v
        continue
    if ('structure_module.layer_norm_z' in k):
        new_k = new_k.replace('structure_module.layer_norm_z', 'StructureModule.pair_layer_norm')
        new_model[new_k] = v
        continue
    if ('structure_module.linear_in' in k):
        new_k = new_k.replace('structure_module.linear_in', 'StructureModule.initial_projection')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.head_weight' in k):
        new_k = new_k.replace('structure_module.ipa.head_weights', 'StructureModule.StructureModuleIteration.InvariantPointAttention.trainable_w')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_q' in k):
        new_k = new_k.replace('structure_module.ipa.linear_q', 'StructureModule.StructureModuleIteration.InvariantPointAttention.q')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_k' in k):
        new_k = new_k.replace('structure_module.ipa.linear_k', 'StructureModule.StructureModuleIteration.InvariantPointAttention.k')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_v' in k):
        new_k = new_k.replace('structure_module.ipa.linear_v', 'StructureModule.StructureModuleIteration.InvariantPointAttention.v')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_q_points' in k):
        new_k = new_k.replace('structure_module.ipa.linear_q_points', 'StructureModule.StructureModuleIteration.InvariantPointAttention.q_points')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_k_points' in k):
        new_k = new_k.replace('structure_module.ipa.linear_k_points', 'StructureModule.StructureModuleIteration.InvariantPointAttention.k_points')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_v_points' in k):
        new_k = new_k.replace('structure_module.ipa.linear_v_points', 'StructureModule.StructureModuleIteration.InvariantPointAttention.v_points')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_b' in k):
        new_k = new_k.replace('structure_module.ipa.linear_b', 'StructureModule.StructureModuleIteration.InvariantPointAttention.rr_kqv_2d')
        new_model[new_k] = v
        continue
    if ('structure_module.ipa.linear_out' in k):
        new_k = new_k.replace('structure_module.ipa.linear_out',
                              'StructureModule.StructureModuleIteration.InvariantPointAttention.final_r')
        new_model[new_k] = v
        continue
    if ('structure_module.layer_norm_ipa' in k):
        new_k = new_k.replace('structure_module.layer_norm_ipa',
                              'StructureModule.StructureModuleIteration.rec_norm')
        new_model[new_k] = v
        continue
    if ('structure_module.transition.layer_norm' in k):
        new_k = new_k.replace('structure_module.transition.layer_norm',
                              'StructureModule.StructureModuleIteration.rec_norm2')
        new_model[new_k] = v
        continue
    if('structure_module.transition.layers.0.linear_1' in k):
        new_k = new_k.replace('structure_module.transition.layers.0.linear_1', 'StructureModule.StructureModuleIteration.transition_r.0')
        new_model[new_k] = v
        continue
    if ('structure_module.transition.layers.0.linear_2' in k):
        new_k = new_k.replace('structure_module.transition.layers.0.linear_2',
                              'StructureModule.StructureModuleIteration.transition_r.2')
        new_model[new_k] = v
        continue
    if ('structure_module.transition.layers.0.linear_3' in k):
        new_k = new_k.replace('structure_module.transition.layers.0.linear_3',
                              'StructureModule.StructureModuleIteration.transition_r.4')
        new_model[new_k] = v
        continue
    if ('structure_module.bb_update.linear' in k):
        new_k = new_k.replace('structure_module.bb_update.linear', 'StructureModule.StructureModuleIteration.backbone_update')
        new_model[new_k] = v
        continue
    if ('structure_module.angle_resnet.linear_in.' in k):
        new_k = new_k.replace('structure_module.angle_resnet.linear_in.', 'StructureModule.StructureModuleIteration.PredictSidechains.s_cur.')
        new_model[new_k] = v
        continue
    if ('structure_module.angle_resnet.linear_initial' in k):
        new_k = new_k.replace('structure_module.angle_resnet.linear_initial', 'StructureModule.StructureModuleIteration.PredictSidechains.s_ini')
        new_model[new_k] = v
        continue
    for i in range(2):
        if('structure_module.angle_resnet.layers.'+str(i)+'.linear_1' in k):
            tmp_idx = 1 + i
            new_k = new_k.replace('structure_module.angle_resnet.layers.'+str(i)+'.linear_1', 'StructureModule.StructureModuleIteration.PredictSidechains.res'+str(tmp_idx)+'.1')
            new_model[new_k] = v
        if ('structure_module.angle_resnet.layers.' + str(i) + '.linear_2' in k):
            new_k = new_k.replace('structure_module.angle_resnet.layers.' + str(i) + '.linear_2',
                                  'StructureModule.StructureModuleIteration.PredictSidechains.res' + str(tmp_idx) + '.3')
            new_model[new_k] = v
    if ('structure_module.angle_resnet.linear_out' in k):
        new_k = new_k.replace('structure_module.angle_resnet.linear_out', 'StructureModule.StructureModuleIteration.PredictSidechains.final.1')
        new_model[new_k] = v
        continue
    if ('aux_heads.plddt.layer_norm' in k):
        new_k = new_k.replace('aux_heads.plddt.layer_norm', 'PredictedLddt.input_layer_norm')
        new_model[new_k] = v
        continue
    if ('aux_heads.plddt.linear_1' in k):
        new_k = new_k.replace('aux_heads.plddt.linear_1', 'PredictedLddt.act_0')
        new_model[new_k] = v
        continue
    if ('aux_heads.plddt.linear_2' in k):
        new_k = new_k.replace('aux_heads.plddt.linear_2', 'PredictedLddt.act_1')
        new_model[new_k] = v
        continue
    if ('aux_heads.plddt.linear_3' in k):
        new_k = new_k.replace('aux_heads.plddt.linear_3', 'PredictedLddt.logits')
        new_model[new_k] = v
        continue
    if ('aux_heads.distogram.linear.' in k):
        new_k = new_k.replace('aux_heads.distogram.linear.', 'Distogram.half_logits.')
        new_model[new_k] = v
        continue
    if ('aux_heads.masked_msa.linear.' in k):
        new_k = new_k.replace('aux_heads.masked_msa.linear.', 'MaskedMsaHead.logits.')
        new_model[new_k] = v
        continue
    if ('aux_heads.pae.linear' in k):
        new_k = new_k.replace('aux_heads.pae.linear', 'PredictedAlignedError.logits')
        new_model[new_k] = v
        continue
    if ('input_embedder.linear_relpos' in k):
        new_k = new_k.replace('input_embedder.linear_relpos', 'InputEmbedder.RecyclingEmbedder.position_activations')
        new_model[new_k] = v
        continue

save_ckpt = sys.argv[2]
model_save_dict = {}
model_save_dict['model'] = new_model
torch.save(model_save_dict, save_ckpt)

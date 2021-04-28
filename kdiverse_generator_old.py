import metric
import numpy as np
import data_generator
import graph_embedding
import lstm_model
import trajectory_generator

np.random.seed(1234567890)


def generate_result(load_from_file, K):
    graph_embedding.get_POI_embeddings(load_from_file=load_from_file)
    lstm_model.get_forward_lstm_model(load_from_file=load_from_file)
    lstm_model.get_backward_lstm_model(load_from_file=load_from_file)

    likability_score = []
    count = 1

    for k, v in data_generator.test_data_dicts_vi[0].items():

        str_k = str(k).split("-")
        poi_start = int(str_k[0])
        poi_end = int(str_k[1])

        _, lstm_order = trajectory_generator.get_prob_in_idx([poi_start, 0, poi_end], 1)
        lstm_rank = np.argsort(lstm_order)

        def get_next_poi(use_freq, rank):
            proposed_poi = 0
            for i in range(len(rank)):
                if i == poi_start or i == poi_end:
                    continue
                if (proposed_poi == poi_start or proposed_poi == poi_end) and (i != poi_start and i != poi_end):
                    proposed_poi = i
                    continue

                if use_freq[i] < use_freq[proposed_poi]:
                    proposed_poi = i
                elif use_freq[i] == use_freq[proposed_poi] and rank[i] < rank[proposed_poi]:
                    proposed_poi = i

            use_freq[proposed_poi] += 1
            return proposed_poi

        use_freq = np.zeros([len(lstm_rank)])
        all_traj = []
        for i in range(K):
            next_poi = get_next_poi(use_freq, lstm_rank)
            new_traj = trajectory_generator.get_sequence(poi_start, next_poi, poi_end, 5, 10)
            for i in range(len(new_traj)):
                use_freq[new_traj[i]] += 1
            all_traj.append(new_traj)

        print("{}/{}".format(count, len(data_generator.test_data_dicts_vi[0])))
        count += 1
        # print([poi_start,poi_end])
        # print(all_traj)
        likability_score_curr = metric.likability_score(v, all_traj)
        likability_score.append(likability_score_curr)
        # print(likability_score_curr)
        print("Avg. upto now: " + str(np.average(likability_score)))

    print("\n")
    print("Final Score - With K = {}".format(K))
    print(np.average(likability_score))

    return

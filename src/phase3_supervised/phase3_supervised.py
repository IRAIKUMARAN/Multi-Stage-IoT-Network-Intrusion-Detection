"""Phase 3 - Supervised classification on FLOW-level data (false-positive reduction).

Match packets -> flows via Flow ID = [src_ip]-[dst_ip]-[src_port]-[dst_port].

Build a second sampled dataset from flows, label them, train a supervised model
(RandomForest / XGBoost / SVM / GNN / deep net) to re-check Phase 2 alerts.

Use labels for TRAINING only, never at test time.
"""

## Comment guidelines: For all the functions explain its working in own words (1-2 lines max)

def match_packet_to_flow(packet_row, flow_df): # change the function when implementing the code
    """Filter flow_df to the same (src_ip, dst_ip, src_port, dst_port)."""
    raise NotImplementedError


def train(X, y): # change the function when implementing the code
    raise NotImplementedError

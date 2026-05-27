"""
customer_segmentation_nba
--------------------------
Modular package for customer segmentation, repurchase prediction
and Next Best Action prescriptive analytics.

Quick start
-----------
    from src.multilang import set_language
    from src.data import load_raw, clean, build_rfm
    from src.segmentation import search_k, segment_names, KMeansClusterAdder
    from src.prediction import make_repurchase_pipeline, evaluate_classifier
    from src.prescriptive import simulate_actions, select_best_action, build_recommendation
    from src.viz import set_corporate_theme

    set_language("pt")   # or "en" (default)
"""

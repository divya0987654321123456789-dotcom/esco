# agent.py
import os
from typing import TypedDict, Dict, List, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json  # <--- FIX: ADD THIS IMPORT

from utils import (
    load_doctr_model,
    extract_text_with_doctr,
    generate_json_via_api,
    apply_scope_preweights_via_api,
    apply_location_preweights,
    load_checklist_items,
    read_company_file,
    save_company_context,
    load_company_context,
    evaluate_checklist_with_api,
    evaluate_checklist_with_consensus,
    export_to_docx,
)

class AgentState(TypedDict):
    rfp_path: str
    rfp_filename: str
    company_files: Dict[str, str]
    checklist_path: str
    
    # Derived
    rfp_full_text: str
    rfp_summary: Dict[str, Any]
    
    preweight_scope_results: Dict[str, Any]
    preweight_location_results: Dict[str, Any]
    combined_preweights: Dict[str, int]
    top_companies: List[str]
    
    company_evaluations: Dict[str, Any]
    report_path: str
    votes: int
    use_reranker: bool
    reranker_top_k: int
    reranker_model_name: str

class BidAnalyzerAgent:
    def __init__(self):
        load_dotenv()
        # Initialize models and clients
        self.doctr_model = load_doctr_model()

        workflow = StateGraph(AgentState)
        workflow.add_node("preprocess_rfp", self._preprocess_rfp)
        workflow.add_node("apply_preweighting", self._apply_preweighting)
        workflow.add_node("evaluate_top_companies", self._evaluate_top_companies)
        workflow.add_node("generate_report", self._generate_report)

        workflow.set_entry_point("preprocess_rfp")
        workflow.add_edge("preprocess_rfp", "apply_preweighting")
        workflow.add_edge("apply_preweighting", "evaluate_top_companies")
        workflow.add_edge("evaluate_top_companies", "generate_report")
        workflow.add_edge("generate_report", END)
        
        self.graph = workflow.compile()

    def _preprocess_rfp(self, state: AgentState) -> Dict[str, Any]:
        print("--- Agent: Pre-processing RFP ---")
        with open(state["rfp_path"], "rb") as f:
            rfp_bytes = f.read()

        rfp_text = extract_text_with_doctr(rfp_bytes, self.doctr_model)
        
        print("--- Agent: Summarizing RFP with API ---")
        prompt = f"""Analyze this RFP and extract key information. Return ONLY valid JSON:
{{
 "project_type": "brief project type", "scope": "brief project scope",
 "summary": "A 2-3 sentence executive summary of the project.",
 "key_requirements": ["requirement1", "requirement2"],
 "due_date": "extracted due date if found, otherwise N/A"
}}
RFP Text (first 8000 chars): {rfp_text[:8000]}"""
        try:
            summary = generate_json_via_api(prompt)
        except Exception:
            summary = {}
        if not isinstance(summary, dict) or not summary:
            print("Warning: API did not return valid JSON. Using basic summary.")
            summary = {"project_type": "General", "scope": rfp_text[:1000]}

        return {"rfp_full_text": rfp_text, "rfp_summary": summary}

    def _apply_preweighting(self, state: AgentState) -> Dict[str, Any]:
        print("--- Agent: Applying Pre-weighting Filters ---")
        scope_results = apply_scope_preweights_via_api(
            state["rfp_summary"], state["rfp_filename"], state["rfp_full_text"]
        )
        location_results = apply_location_preweights(
            state["rfp_full_text"], state["rfp_summary"], state["rfp_filename"]
        )
        
        c1 = scope_results.get("preweights", {})
        c2 = location_results.get("preweights", {})
        companies = ["IKIO", "IKIO ENERGY", "IKIO ENERGY"]
        combined = {c: int(c1.get(c, 0)) + int(c2.get(c, 0)) for c in companies}
        
        # Select top 2 companies for detailed evaluation
        top_companies = sorted(companies, key=lambda x: combined[x], reverse=True)[:2]
        print(f"--- Agent: Top companies selected: {top_companies} ---")
        
        return {
            "preweight_scope_results": scope_results,
            "preweight_location_results": location_results,
            "combined_preweights": combined,
            "top_companies": top_companies,
        }

    def _evaluate_top_companies(self, state: AgentState) -> Dict[str, Any]:
        print("--- Agent: Performing Detailed Evaluation on Top Companies (Combined Checklist Only) ---")
        combined_items = load_checklist_items(state["checklist_path"])
        evaluations: Dict[str, Any] = {}

        # Evaluate only top 2 after preweighting as per business rule
        top_companies = state.get("top_companies", ["IKIO", "IKIO ENERGY"])[:2]
        votes = int(state.get("votes", 1) or 1)
        use_reranker = bool(state.get("use_reranker", False))
        reranker_top_k = int(state.get("reranker_top_k", 2) or 2)
        reranker_model_name = str(state.get("reranker_model_name", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2")
        for name in top_companies:
            company_file_path = state["company_files"].get(name)
            context = ""
            if company_file_path:
                context = read_company_file(company_file_path, self.doctr_model)
                save_company_context(name, context)
            else:
                context = load_company_context(name)

            # Even if context is empty, we can still run RFP-only and Combined modes; Company-only will likely be partial.
            print(f"--- Agent: Evaluating {name} (Combined) ---")
            if votes > 1:
                combined_items_eval = evaluate_checklist_with_consensus(
                    name,
                    state["rfp_summary"],
                    context,
                    combined_items,
                    state["rfp_full_text"],
                    mode="combined",
                    votes=votes,
                    temperature=0.3,
                    use_reranker=use_reranker,
                    reranker_top_k=reranker_top_k,
                    reranker_model_name=reranker_model_name,
                )
            else:
                combined_items_eval = evaluate_checklist_with_api(
                    name,
                    state["rfp_summary"],
                    context,
                    combined_items,
                    state["rfp_full_text"],
                    mode="combined",
                    temperature=0.0,
                    use_reranker=use_reranker,
                    reranker_top_k=reranker_top_k,
                    reranker_model_name=reranker_model_name,
                )
            comb_total = sum(it.get("score", 0) for it in combined_items_eval)
            comb_max = len(combined_items_eval) * 2.0
            comb_pct = round((comb_total / comb_max) * 100, 1) if comb_max else 0
            combined_block = {
                "complianceAssessment": combined_items_eval,
                "finalScoring": {
                    "totalScore": round(comb_total, 2),
                    "overallCompliancePercentage": comb_pct,
                    "recommendation": "Go" if comb_pct >= 85 else "Conditional Go" if comb_pct >= 70 else "No-Go",
                },
            }

            evaluations[name] = {"combined": combined_block}

        return {"company_evaluations": evaluations}

    def _generate_report(self, state: AgentState) -> Dict[str, Any]:
        print("--- Agent: Generating DOCX Report ---")
        report_path = "Bid_Analysis_Report.docx"
        export_to_docx(state, report_path)
        print(f"--- Agent: Report saved to {report_path} ---")
        return {"report_path": report_path}

    def run(
        self,
        rfp_path: str,
        rfp_filename: str,
        company_files: Dict[str, str],
        checklist_path: str | None = None,
        votes: int | None = 1,
        use_reranker: bool | None = False,
        reranker_top_k: int | None = 2,
        reranker_model_name: str | None = "all-MiniLM-L6-v2",
    ) -> Dict[str, Any]:
        initial_state: AgentState = {
            "rfp_path": rfp_path,
            "rfp_filename": rfp_filename,
            "company_files": company_files,
            "checklist_path": checklist_path or "checklist.xlsx",
            "votes": int(votes or 1),
            "use_reranker": bool(use_reranker or False),
            "reranker_top_k": int(reranker_top_k or 2),
            "reranker_model_name": str(reranker_model_name or "all-MiniLM-L6-v2"),
        }
        return self.graph.invoke(initial_state)

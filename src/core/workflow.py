"""
LangGraph workflow for Agentic RAG with durable execution.
"""
import logging
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.query_analyzer import QueryAnalyzerAgent, QueryAnalysis
from src.agents.retriever import RetrievalAgent, RetrievalResult
from src.agents.synthesizer import SynthesisAgent, SynthesisResult
from src.agents.evaluator import EvaluationAgent, EvaluationResult
from config.config import config

logger = logging.getLogger(__name__)


# Define the workflow state
class AgenticRAGState(TypedDict):
    """State for the Agentic RAG workflow."""
    # Input
    query: str

    # Query Analysis
    query_analysis: Optional[QueryAnalysis]

    # Retrieval
    retrieval_result: Optional[RetrievalResult]
    retrieval_attempt: int

    # Synthesis
    synthesis_result: Optional[SynthesisResult]
    synthesis_attempt: int

    # Evaluation
    evaluation_result: Optional[EvaluationResult]

    # Final output
    final_answer: Optional[str]
    final_sources: Optional[List[str]]

    # Workflow control
    error: Optional[str]
    max_retries: int
    workflow_step: str


class AgenticRAGWorkflow:
    """
    LangGraph workflow for orchestrating the Agentic RAG pipeline.
    """

    def __init__(
        self,
        query_analyzer: Optional[QueryAnalyzerAgent] = None,
        retriever: Optional[RetrievalAgent] = None,
        synthesizer: Optional[SynthesisAgent] = None,
        evaluator: Optional[EvaluationAgent] = None,
        enable_checkpointing: bool = True
    ):
        """
        Initialize the workflow.

        Args:
            query_analyzer: QueryAnalyzerAgent instance
            retriever: RetrievalAgent instance
            synthesizer: SynthesisAgent instance
            evaluator: EvaluationAgent instance
            enable_checkpointing: Whether to enable durable execution
        """
        # Initialize agents
        self.query_analyzer = query_analyzer or QueryAnalyzerAgent()
        self.retriever = retriever or RetrievalAgent()
        self.synthesizer = synthesizer or SynthesisAgent()
        self.evaluator = evaluator or EvaluationAgent()

        # Setup checkpointing for durable execution
        self.checkpointer = None
        if enable_checkpointing and config.langgraph.enable_durable_execution:
            checkpoint_path = str(config.langgraph.checkpoint_path)
            self.checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
            logger.info(f"Checkpointing enabled at {checkpoint_path}")

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

        logger.info("AgenticRAGWorkflow initialized successfully")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            StateGraph instance
        """
        workflow = StateGraph(AgenticRAGState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        workflow.add_node("finalize", self._finalize_node)

        # Define edges
        workflow.set_entry_point("analyze_query")

        # analyze_query -> retrieve_documents
        workflow.add_edge("analyze_query", "retrieve_documents")

        # retrieve_documents -> synthesize_answer or END (if retrieval failed)
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._should_continue_after_retrieval,
            {
                "synthesize": "synthesize_answer",
                "end": END
            }
        )

        # synthesize_answer -> evaluate_answer
        workflow.add_edge("synthesize_answer", "evaluate_answer")

        # evaluate_answer -> finalize or retry synthesis
        workflow.add_conditional_edges(
            "evaluate_answer",
            self._should_retry_or_finalize,
            {
                "retry_synthesis": "synthesize_answer",
                "retry_retrieval": "retrieve_documents",
                "finalize": "finalize"
            }
        )

        # finalize -> END
        workflow.add_edge("finalize", END)

        logger.info("Workflow graph built successfully")
        return workflow

    async def _analyze_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Analyze the user query."""
        logger.info(f"Node: Analyzing query - '{state['query'][:100]}...'")

        state["workflow_step"] = "analyze_query"

        try:
            query_analysis = await self.query_analyzer.analyze_query(state["query"])
            state["query_analysis"] = query_analysis
            logger.info(f"Query analysis complete: intent={query_analysis.intent}")
        except Exception as e:
            logger.error(f"Error in analyze_query_node: {e}")
            state["error"] = f"Query analysis failed: {str(e)}"

        return state

    async def _retrieve_documents_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Retrieve relevant documents."""
        logger.info("Node: Retrieving documents")

        state["workflow_step"] = "retrieve_documents"
        state["retrieval_attempt"] = state.get("retrieval_attempt", 0) + 1

        try:
            retrieval_result = await self.retriever.retrieve(
                query=state["query"],
                query_analysis=state.get("query_analysis")
            )
            state["retrieval_result"] = retrieval_result
            logger.info(f"Retrieved {retrieval_result.total_retrieved} documents")
        except Exception as e:
            logger.error(f"Error in retrieve_documents_node: {e}")
            state["error"] = f"Retrieval failed: {str(e)}"

        return state

    async def _synthesize_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Synthesize answer from retrieved documents."""
        logger.info("Node: Synthesizing answer")

        state["workflow_step"] = "synthesize_answer"
        state["synthesis_attempt"] = state.get("synthesis_attempt", 0) + 1

        try:
            synthesis_result = await self.synthesizer.synthesize(
                query=state["query"],
                retrieval_result=state["retrieval_result"]
            )
            state["synthesis_result"] = synthesis_result
            logger.info("Answer synthesized successfully")
        except Exception as e:
            logger.error(f"Error in synthesize_answer_node: {e}")
            state["error"] = f"Synthesis failed: {str(e)}"

        return state

    async def _evaluate_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Evaluate the synthesized answer."""
        logger.info("Node: Evaluating answer")

        state["workflow_step"] = "evaluate_answer"

        try:
            evaluation_result = await self.evaluator.evaluate(
                query=state["query"],
                synthesis_result=state["synthesis_result"],
                retrieval_result=state["retrieval_result"]
            )
            state["evaluation_result"] = evaluation_result
            logger.info(f"Evaluation complete: score={evaluation_result.overall_score:.2f}")
        except Exception as e:
            logger.error(f"Error in evaluate_answer_node: {e}")
            state["error"] = f"Evaluation failed: {str(e)}"

        return state

    async def _finalize_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Finalize the response."""
        logger.info("Node: Finalizing response")

        state["workflow_step"] = "finalize"

        if state.get("synthesis_result"):
            state["final_answer"] = state["synthesis_result"].answer
            state["final_sources"] = state["synthesis_result"].sources_used
        else:
            state["final_answer"] = "I apologize, but I couldn't generate an answer."
            state["final_sources"] = []

        logger.info("Response finalized")
        return state

    def _should_continue_after_retrieval(self, state: AgenticRAGState) -> str:
        """Decide whether to continue after retrieval."""
        retrieval_result = state.get("retrieval_result")

        if state.get("error") or not retrieval_result:
            logger.warning("Retrieval failed, ending workflow")
            return "end"

        if retrieval_result.total_retrieved == 0:
            logger.warning("No documents retrieved, ending workflow")
            # Set a default error state
            state["final_answer"] = "I couldn't find relevant documents to answer your question."
            state["final_sources"] = []
            return "end"

        return "synthesize"

    def _should_retry_or_finalize(self, state: AgenticRAGState) -> str:
        """Decide whether to retry or finalize."""
        evaluation_result = state.get("evaluation_result")
        max_retries = state.get("max_retries", config.agent.max_retries)
        synthesis_attempt = state.get("synthesis_attempt", 0)
        retrieval_attempt = state.get("retrieval_attempt", 0)

        # If there's an error, finalize
        if state.get("error"):
            logger.warning("Error detected, finalizing")
            return "finalize"

        # If no evaluation result, finalize
        if not evaluation_result:
            return "finalize"

        # If approved, finalize
        if evaluation_result.approved:
            logger.info("Answer approved, finalizing")
            return "finalize"

        # If needs regeneration and under retry limit
        if evaluation_result.needs_regeneration:
            if synthesis_attempt < max_retries:
                logger.info(f"Retrying synthesis (attempt {synthesis_attempt + 1}/{max_retries})")
                return "retry_synthesis"
            elif retrieval_attempt < max_retries:
                logger.info(f"Retrying retrieval (attempt {retrieval_attempt + 1}/{max_retries})")
                return "retry_retrieval"

        # Max retries reached or no regeneration needed
        logger.info("Finalizing (max retries reached or no regeneration needed)")
        return "finalize"

    async def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the workflow for a query.

        Args:
            query: User query
            thread_id: Optional thread ID for checkpointing
            max_retries: Maximum number of retries (defaults to config)

        Returns:
            Dictionary with final answer and metadata
        """
        logger.info(f"Starting workflow for query: '{query[:100]}...'")

        # Initialize state
        initial_state: AgenticRAGState = {
            "query": query,
            "query_analysis": None,
            "retrieval_result": None,
            "retrieval_attempt": 0,
            "synthesis_result": None,
            "synthesis_attempt": 0,
            "evaluation_result": None,
            "final_answer": None,
            "final_sources": None,
            "error": None,
            "max_retries": max_retries or config.agent.max_retries,
            "workflow_step": "init"
        }

        # Configure thread
        config_dict = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            # Run the workflow
            final_state = await self.app.ainvoke(initial_state, config=config_dict)

            logger.info("Workflow completed successfully")

            return {
                "answer": final_state.get("final_answer"),
                "sources": final_state.get("final_sources"),
                "query_analysis": final_state.get("query_analysis"),
                "evaluation": final_state.get("evaluation_result"),
                "retrieval_attempts": final_state.get("retrieval_attempt", 0),
                "synthesis_attempts": final_state.get("synthesis_attempt", 0),
                "error": final_state.get("error")
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "answer": "An error occurred while processing your query.",
                "sources": [],
                "error": str(e)
            }

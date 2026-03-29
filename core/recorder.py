import json
from collections import defaultdict

from sqlalchemy import func,select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.classifier import Classification
from core.config import AppConfig
from models.request import RequestRecord, prompt_hash
from router.proxy import ProxyResult

_TOP_KEYWORDS = 10
_MIN_REQUESTS_FOR_RECALIBRATION = 100


def _extract_keywords(prompt:str) -> list[str]:
    stop_words = frozenset({
       "the", "a", "an", "is", "in", "it", "of", "to", "and", "or",
        "for", "with", "this", "that", "how", "what", "why", "when",
        "where", "which", "do", "does", "can", "could", "would", "should",
        "i", "me", "my", "you", "your", "we", "our", "they", "their",
        "be", "been", "being", "have", "has", "had", "was", "were", "are",
        "will", "just", "please", "make", "get", "use",
    })

    words = [
        w.lower().strip(".,!?;:\"'()")
        for w in prompt.split()
    ]
    keywords = [w for w in words if w and w not in stop_words and len(w) > 2]
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:_TOP_KEYWORDS]


def _is_misclassified(estimated: int, actual:int, ratio: float) -> bool:
    return actual > estimated * ratio


def _is_overprovisioned(estimated:int, actual:int, ratio:float) -> bool:
    return actual < estimated * ratio

class Recorder:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        config: AppConfig
    )-> None:
        self._session_factory = session_factory
        self._config = config

    async def record(
            self,
            prompt:str,
            classification:Classification,
            result:ProxyResult
    )->None:
        thresholds = self._config.thresholds
        actual_output = result.response.output_tokens
        estimated_output = classification.max_tokens

        misclassified = _is_misclassified(
            estimated_output,
            actual_output,
            thresholds.misclassification_ratio
        )
        keywords = _extract_keywords(prompt)

        record = RequestRecord(
            prompt_hash=prompt_hash(prompt),
            keywords=json.dumps(keywords),
            task_type=classification.task_type,
            routing_confidence=classification.confidence,
            estimated_output_tokens=estimated_output,
            actual_input_tokens=result.response.input_tokens,
            actual_output_tokens=actual_output,
            max_tokens_set=estimated_output,
            model_used=result.response.model,
            provider_used=result.provider_name,
            was_fallback=result.was_fallback,
            fallback_reason=str(result.fallback_reason) if result.fallback_reason else None,
            was_upgraded=result.was_upgraded,
            response_time_ms=result.response_time_ms,
            was_misclassified=misclassified
        )

        async with self._session_factory() as session:
            session.add(record)
            await session.commit()

        if misclassified:
            await self._recalibrate(classification.task_type, result.provider_name)
    
    async def _recalibrate(self,task_type:str, provider_name:str)-> None:
        """
        After 100+ requests for a task_type+provider pair, recompute the average
        actual_output_tokens and flag outlier records as misclassified.
        Recalibration is per-provider because tokenizers differ across providers.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.count(RequestRecord.id),
                    func.avg(RequestRecord.actual_output_tokens),
                ).where(
                    RequestRecord.task_type == task_type,
                    RequestRecord.provider_used == provider_name
                )
            )
            row = result.one()
            count: int = row[0]
            avg_actual:float | None = row[1]

            if count < _MIN_REQUESTS_FOR_RECALIBRATION or avg_actual is None:
                return
            
            #re-flagging records whose estimated tokens are far from the new average

            thresholds = self._config.thresholds
            await session.execute(
                update(RequestRecord)
                .where(
                    RequestRecord.task_type == task_type,
                    RequestRecord.provider_used == provider_name
                )
                .values(
                    was_misclassified=(
                        RequestRecord.actual_output_tokens>RequestRecord.estimated_output_tokens * thresholds.misclassification_ratio
                    )
                )
            )
            await session.commit()

    async def stats(self) -> dict:
        async with self._session_factory() as session:
            total_result = await session.execute(select(func.count(RequestRecord.id)))
            total:int = total_result.scalar() or 0

            if total == 0:
                return _empty_stats()
            
            #routing accuracy
            misclassified_result = await session.execute(
                select(func.count(RequestRecord.id)).where(
                    RequestRecord.was_misclassified == True
                )
            )
            misclassified_count: int = misclassified_result.scalar() or 0
            routing_accuracy = round(1- misclassified_count / total, 4)

            #model distribution
            model_rows = await session.execute(
                select(RequestRecord.task_type, func.count(RequestRecord.id))
            .group_by(RequestRecord.task_type)
            )

            #map task_type -> tier via a lightweight lookup
            from core.classifier import _TASK_TIERS # local import to avoid circular
            tier_counts: dict[str,int] = defaultdict(int)
            for task_type, count in model_rows:
                tier = _TASK_TIERS.get(task_type, "medium")
                tier_counts[str(tier)] += count
            model_distribution = {
                tier: round(count / total, 4)
                for tier,count in tier_counts.items()
            }

            #provider distribution
            provider_rows = await session.execute(
                select(RequestRecord.provider_used, func.count(RequestRecord.id))
                .group_by(RequestRecord.provider_used)
            )
            provider_distribution = {
                provider: round(count / total, 4)
                for provider, count in provider_rows
            }

            #avg tokens saved vs always routing to large with max cap (900)
            avg_tokens_result = await session.execute(
                select(func.avg(RequestRecord.max_tokens_set))
            )
            avg_tokens_set:float = avg_tokens_result.scalar() or 0.0
            avg_tokens_saved = round(900 - avg_tokens_set, 2)

            #fallback + upgrade rates
            fallback_result = await session.execute(
                select(func.count(RequestRecord.id)).where(
                    RequestRecord.was_fallback == True
                )
            )
            fallback_count: int = fallback_result.scalar() or 0

            upgrade_result =await session.execute(
                select(func.count(RequestRecord.id)).where(
                    RequestRecord.was_upgraded == True
                )
            )
            upgrade_count: int = upgrade_result.scalar() or 0

            #most misclassified task type
            misclass_rows = await session.execute(
                select(RequestRecord.task_type, func.count(RequestRecord.id))
                .where(RequestRecord.was_misclassified == True)
                .group_by(RequestRecord.task_type)
                .order_by(func.count(RequestRecord.id).desc())
                .limit(1)
            )
            most_misclassified_row = misclass_rows.first()
            most_misclassified_type = most_misclassified_row[0] if most_misclassified_row else None

            #avg response time by tier
            time_rows = await session.execute(
                select(RequestRecord.task_type, func.avg(RequestRecord.response_time_ms))
                .group_by(RequestRecord.task_type)
            )
            tier_times: dict[str,list[float]] = defaultdict(list)
            for task_type,avg_ms in time_rows:
                tier = str(_TASK_TIERS.get(task_type,"medium")) #type: ignore[call-overload]
                tier_times[tier].append(avg_ms or 0.0)
            avg_response_time_ms = {
                tier: round(sum(times) / len(times),2)
                for tier,times in tier_times.items()
            }

        return {
            "total_requests": total,
            "routing_accuracy":routing_accuracy,
            "model_distribution":model_distribution,
            "provider_distribution":provider_distribution,
            "avg_tokens_saved_vs_always_large":avg_tokens_saved,
            "fallback_rate":round(fallback_count / total, 4),
            "upgrade_rate":round(upgrade_count / total, 4),
            "most_misclassified_type":most_misclassified_type,
            "avg_response_time_ms": avg_response_time_ms
        }
    

def _empty_stats()-> dict:
    return {
            "total_requests": 0,
            "routing_accuracy":0.0,
            "model_distribution":{},
            "provider_distribution":{},
            "avg_tokens_saved_vs_always_large":0.0,
            "fallback_rate":0.0,
            "upgrade_rate":0.0,
            "most_misclassified_type":None,
            "avg_response_time_ms": {}
    }





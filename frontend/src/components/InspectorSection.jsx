import { useEffect, useRef, useState } from "react";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";

export function InspectorSection({ id, title, icon: Icon, expanded, onToggle, summary, errorCount = 0, children }) {
  const sectionRef = useRef(null);
  const dragRef = useRef(null);
  const [height, setHeight] = useState(null);
  const [measuredHeight, setMeasuredHeight] = useState(120);
  const minHeight = 120;
  const maxHeight = 900;
  const hasErrors = errorCount > 0;

  useEffect(() => {
    const section = sectionRef.current;
    if (!section || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      const measured = Math.round(section.getBoundingClientRect().height);
      setMeasuredHeight((previous) => previous === measured ? previous : measured);
    });
    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  function resize(nextHeight) {
    setHeight(Math.round(Math.min(maxHeight, Math.max(minHeight, nextHeight))));
  }

  function handleResizeKeyDown(event) {
    const current = height ?? sectionRef.current?.getBoundingClientRect().height ?? minHeight;
    const step = event.shiftKey ? 48 : 16;
    if (event.key === "ArrowUp") resize(current - step);
    else if (event.key === "ArrowDown") resize(current + step);
    else if (event.key === "Home") resize(minHeight);
    else if (event.key === "End") resize(maxHeight);
    else return;
    event.preventDefault();
  }

  return (
    <section
      id={`${id}-section`}
      ref={sectionRef}
      className={`inspector-section${expanded ? " inspector-section--open" : ""}${height !== null ? " inspector-section--resized" : ""}${hasErrors ? " inspector-section--error" : ""}`}
      style={expanded && height !== null ? { height: `${height}px` } : undefined}
      aria-labelledby={`${id}-toggle`}
    >
      <button
        id={`${id}-toggle`}
        className="inspector-section__toggle"
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-controls={`${id}-content`}
      >
        <span className="inspector-section__heading">
          <span className="inspector-section__title">
            <Icon aria-hidden="true" size={17} stroke={1.8} />
            {title}
          </span>
          {!expanded && summary ? <span className="inspector-section__summary">{summary}</span> : null}
        </span>
        <span className="inspector-section__indicators">
          {hasErrors ? <span className="inspector-section__error-count">{errorCount} {errorCount === 1 ? "error" : "errors"}</span> : null}
          {expanded ? (
            <IconChevronUp aria-hidden="true" size={17} />
          ) : (
            <IconChevronDown aria-hidden="true" size={17} />
          )}
        </span>
      </button>
      <div className="inspector-section__content" id={`${id}-content`} hidden={!expanded}>
        {children}
      </div>
      {expanded ? (
        <div
          className="inspector-section__resizer"
          role="separator"
          tabIndex={0}
          aria-label={`Resize ${title} panel vertically`}
          aria-orientation="horizontal"
          aria-controls={`${id}-content`}
          aria-valuemin={minHeight}
          aria-valuemax={maxHeight}
          aria-valuenow={height ?? measuredHeight}
          aria-valuetext={height === null ? "Automatic height" : `${height} pixels`}
          title={`Drag to resize ${title}; double-click to reset`}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            dragRef.current = {
              startY: event.clientY,
              startHeight: sectionRef.current.getBoundingClientRect().height,
            };
            event.currentTarget.setPointerCapture(event.pointerId);
            event.preventDefault();
          }}
          onPointerMove={(event) => {
            if (!dragRef.current) return;
            resize(dragRef.current.startHeight + event.clientY - dragRef.current.startY);
          }}
          onPointerUp={() => { dragRef.current = null; }}
          onPointerCancel={() => { dragRef.current = null; }}
          onKeyDown={handleResizeKeyDown}
          onDoubleClick={() => setHeight(null)}
        >
          <span aria-hidden="true" />
        </div>
      ) : null}
    </section>
  );
}

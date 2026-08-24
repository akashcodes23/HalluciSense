import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-all duration-150 cursor-pointer disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]",
  {
    variants: {
      variant: {
        default:
          "bg-[var(--primary)] text-white hover:bg-[var(--primary-hover)] shadow-[var(--shadow-sm)]",
        secondary:
          "bg-[var(--surface)] text-[var(--text-primary)] border border-[var(--border)] hover:bg-[var(--surface-hover)] hover:border-[var(--border-hover)]",
        outline:
          "border border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)] hover:border-[var(--border-hover)]",
        ghost:
          "text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]",
        destructive:
          "bg-[var(--hallucination-soft)] text-[var(--hallucination)] border border-[var(--hallucination-border)] hover:bg-[var(--hallucination)] hover:text-white",
        verified:
          "bg-[var(--verified-soft)] text-[var(--verified)] border border-[var(--verified-border)] hover:bg-[var(--verified)] hover:text-white",
        link:
          "text-[var(--primary)] underline-offset-4 hover:underline p-0 h-auto",
      },
      size: {
        default: "h-9 px-3.5 rounded-[var(--radius)]",
        sm: "h-7 px-2.5 text-xs rounded-[var(--radius-sm)]",
        lg: "h-10 px-5 rounded-[var(--radius-md)]",
        xl: "h-11 px-6 text-base rounded-[var(--radius-md)]",
        icon: "h-8 w-8 rounded-[var(--radius)]",
        "icon-sm": "h-7 w-7 rounded-[var(--radius-sm)]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
